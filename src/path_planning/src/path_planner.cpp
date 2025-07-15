#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;

bool VERBOSE_UNNECESSARY_THINGS = false;

struct AStarNode
{
    int x, y;
    float g, h;
    AStarNode *parent;

    float f() const { return g + h; }

    bool operator>(const AStarNode &other) const
    {
        return f() > other.f();
    }
};

class AStarPlanner : public rclcpp::Node
{
public:
    AStarPlanner() : Node("a_star_planner")
    {
        costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/costmap", 10, std::bind(&AStarPlanner::costmap_callback, this, _1));

        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_point", 10, std::bind(&AStarPlanner::goal_callback, this, _1));

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/sm_planned_path", 10);
    }

private:
    nav_msgs::msg::OccupancyGrid::SharedPtr costmap_;
    geometry_msgs::msg::PoseStamped goal_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    std::pair<int, int> world_to_map(double wx, double wy)
    {
        int mx = static_cast<int>((wx - costmap_->info.origin.position.x) / costmap_->info.resolution);
        int my = static_cast<int>((wy - costmap_->info.origin.position.y) / costmap_->info.resolution);
        return {mx, my};
    }

    geometry_msgs::msg::PoseStamped map_to_world(int mx, int my)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = costmap_->header.frame_id;
        pose.pose.position.x = costmap_->info.origin.position.x + (mx + 0.5) * costmap_->info.resolution;
        pose.pose.position.y = costmap_->info.origin.position.y + (my + 0.5) * costmap_->info.resolution;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.w = 1.0;
        return pose;
    }

    bool is_valid(int x, int y)
    {
        int width = costmap_->info.width;
        int height = costmap_->info.height;
        return x >= 0 && y >= 0 && x < width && y < height;
    }

    int index(int x, int y)
    {
        return y * costmap_->info.width + x;
    }

    float heuristic(int x1, int y1, int x2, int y2)
    {
        return std::hypot(x1 - x2, y1 - y2);
    }

    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!costmap_)
        {
            RCLCPP_WARN(this->get_logger(), "No costmap received yet.");
            return;
        }

        goal_ = *msg;

        int start_x = 150;
        int start_y = 150;
        // Change to actual robot pose
        auto [goal_x, goal_y] = world_to_map(goal_.pose.position.x, goal_.pose.position.y);

        if (VERBOSE_UNNECESSARY_THINGS) RCLCPP_INFO(this->get_logger(), "Planning from (%d, %d) to (%d, %d)", start_x, start_y, goal_x, goal_y);

        auto raw_path = a_star(start_x, start_y, goal_x, goal_y);
        raw_path.header.stamp = this->now();
        raw_path.header.frame_id = costmap_->header.frame_id;

        // Smooth the path
        auto smoothed_poses = smooth_path(raw_path.poses);
        raw_path.poses = smoothed_poses;

        path_pub_->publish(raw_path);
    }

    void costmap_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        costmap_ = msg;
    }

    nav_msgs::msg::Path a_star(int start_x, int start_y, int goal_x, int goal_y)
    {
        nav_msgs::msg::Path path_msg;

        auto cmp = [](AStarNode *a, AStarNode *b)
        { return *a > *b; };
        std::priority_queue<AStarNode *, std::vector<AStarNode *>, decltype(cmp)> open(cmp);

        std::vector<std::vector<bool>> closed(costmap_->info.width, std::vector<bool>(costmap_->info.height, false));

        AStarNode *start = new AStarNode{start_x, start_y, 0.0f, heuristic(start_x, start_y, goal_x, goal_y), nullptr};
        open.push(start);

        const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

        while (!open.empty())
        {
            AStarNode *current = open.top();
            open.pop();

            if (current->x == goal_x && current->y == goal_y)
            {
                AStarNode *node = current;
                while (node)
                {
                    path_msg.poses.push_back(map_to_world(node->x, node->y));
                    node = node->parent;
                }
                std::reverse(path_msg.poses.begin(), path_msg.poses.end());
                break;
            }

            if (closed[current->x][current->y])
                continue;
            closed[current->x][current->y] = true;

            for (int i = 0; i < 8; ++i)
            {
                int nx = current->x + dx[i];
                int ny = current->y + dy[i];

                if (!is_valid(nx, ny))
                    continue;

                int cost = costmap_->data[index(nx, ny)];
                if (cost < 0 || cost > 50)
                    continue;

                float g_new = current->g + std::hypot(dx[i], dy[i]) + cost / 100.0;
                float h_new = heuristic(nx, ny, goal_x, goal_y);

                AStarNode *neighbor = new AStarNode{nx, ny, g_new, h_new, current};
                open.push(neighbor);
            }
        }

        return path_msg;
    }
    std::vector<geometry_msgs::msg::PoseStamped> smooth_path(const std::vector<geometry_msgs::msg::PoseStamped> &path, float alpha = 0.009, float beta = 0.4, float tolerance = 0.00001)
    {
        auto new_path = path;
        float change = tolerance;
        int max_iter = 1000;
        int n = path.size();

        for (int iter = 0; iter < max_iter && change >= tolerance; ++iter)
        {
            change = 0.0;

            for (int i = 1; i < n - 1; ++i)
            {
                float x_old = new_path[i].pose.position.x;
                float y_old = new_path[i].pose.position.y;

                float x_data = path[i].pose.position.x;
                float y_data = path[i].pose.position.y;

                float x_prev = new_path[i - 1].pose.position.x;
                float y_prev = new_path[i - 1].pose.position.y;

                float x_next = new_path[i + 1].pose.position.x;
                float y_next = new_path[i + 1].pose.position.y;

                new_path[i].pose.position.x += alpha * (x_data - x_old) + beta * (x_prev + x_next - 2 * x_old);
                new_path[i].pose.position.y += alpha * (y_data - y_old) + beta * (y_prev + y_next - 2 * y_old);

                change += (x_old - new_path[i].pose.position.x > 0 ? x_old - new_path[i].pose.position.x : new_path[i].pose.position.x - x_old) +
                          (y_old - new_path[i].pose.position.y > 0 ? y_old - new_path[i].pose.position.y : new_path[i].pose.position.y - y_old);
            }
        }

        return new_path;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AStarPlanner>());
    rclcpp::shutdown();
    return 0;
}
