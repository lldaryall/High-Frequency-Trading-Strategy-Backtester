#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <string>
#include <cstdint>

namespace flashback {

// Order side enumeration
enum class OrderSide : int8_t {
    BUY = 1,
    SELL = -1
};

// Time in force enumeration
enum class TimeInForce : int8_t {
    DAY = 0,
    IOC = 1,  // Immediate or Cancel
    FOK = 2   // Fill or Kill
};

// Order status
enum class OrderStatus : int8_t {
    PENDING = 0,
    PARTIALLY_FILLED = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4
};

// Fill structure
struct Fill {
    std::string order_id;
    double price;
    int64_t quantity;
    
    Fill(const std::string& oid, double p, int64_t q) 
        : order_id(oid), price(p), quantity(q) {}
};

// Order structure
struct Order {
    std::string order_id;
    OrderSide side;
    double price;
    int64_t quantity;
    int64_t filled_quantity;
    TimeInForce tif;
    OrderStatus status;
    int64_t timestamp;
    
    Order(const std::string& oid, OrderSide s, double p, int64_t q, TimeInForce t, int64_t ts)
        : order_id(oid), side(s), price(p), quantity(q), filled_quantity(0), 
          tif(t), status(OrderStatus::PENDING), timestamp(ts) {}
    
    int64_t remaining_quantity() const {
        return quantity - filled_quantity;
    }
    
    bool is_active() const {
        return status == OrderStatus::PENDING || status == OrderStatus::PARTIALLY_FILLED;
    }
};

// Price level structure
struct PriceLevel {
    double price;
    std::queue<std::shared_ptr<Order>> orders;
    int64_t total_quantity;
    
    PriceLevel(double p) : price(p), total_quantity(0) {}
};

// Thread-safe matching engine
class MatchEngine {
private:
    // Order books: price -> PriceLevel
    std::map<double, std::unique_ptr<PriceLevel>, std::greater<double>> bid_book_;
    std::map<double, std::unique_ptr<PriceLevel>, std::less<double>> ask_book_;
    
    // Order lookup: order_id -> Order*
    std::map<std::string, std::shared_ptr<Order>> orders_;
    
    // Fills generated during matching
    std::vector<Fill> fills_;
    
    // Thread safety
    mutable std::mutex mutex_;
    
    // Order ID counter for internal orders
    std::atomic<int64_t> order_counter_{0};
    
    // Helper methods
    void add_order_to_bid_book(std::shared_ptr<Order> order) {
        double price = order->price;
        
        if (bid_book_.find(price) == bid_book_.end()) {
            bid_book_[price] = std::make_unique<PriceLevel>(price);
        }
        
        bid_book_[price]->orders.push(order);
        bid_book_[price]->total_quantity += order->remaining_quantity();
        orders_[order->order_id] = order;
    }
    
    void add_order_to_ask_book(std::shared_ptr<Order> order) {
        double price = order->price;
        
        if (ask_book_.find(price) == ask_book_.end()) {
            ask_book_[price] = std::make_unique<PriceLevel>(price);
        }
        
        ask_book_[price]->orders.push(order);
        ask_book_[price]->total_quantity += order->remaining_quantity();
        orders_[order->order_id] = order;
    }
    
    void remove_order_from_bid_book(const std::string& order_id) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;
        
        auto order = it->second;
        double price = order->price;
        
        auto book_it = bid_book_.find(price);
        if (book_it != bid_book_.end()) {
            // Remove from queue
            std::queue<std::shared_ptr<Order>> new_queue;
            while (!book_it->second->orders.empty()) {
                auto front_order = book_it->second->orders.front();
                book_it->second->orders.pop();
                if (front_order->order_id != order_id) {
                    new_queue.push(front_order);
                }
            }
            book_it->second->orders = std::move(new_queue);
            book_it->second->total_quantity -= order->remaining_quantity();
            
            // Remove price level if empty
            if (book_it->second->orders.empty()) {
                bid_book_.erase(book_it);
            }
        }
        
        orders_.erase(it);
    }
    
    void remove_order_from_ask_book(const std::string& order_id) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;
        
        auto order = it->second;
        double price = order->price;
        
        auto book_it = ask_book_.find(price);
        if (book_it != ask_book_.end()) {
            // Remove from queue
            std::queue<std::shared_ptr<Order>> new_queue;
            while (!book_it->second->orders.empty()) {
                auto front_order = book_it->second->orders.front();
                book_it->second->orders.pop();
                if (front_order->order_id != order_id) {
                    new_queue.push(front_order);
                }
            }
            book_it->second->orders = std::move(new_queue);
            book_it->second->total_quantity -= order->remaining_quantity();
            
            // Remove price level if empty
            if (book_it->second->orders.empty()) {
                ask_book_.erase(book_it);
            }
        }
        
        orders_.erase(it);
    }
    
    void process_fill(std::shared_ptr<Order> order, double fill_price, int64_t fill_qty) {
        order->filled_quantity += fill_qty;
        
        if (order->filled_quantity >= order->quantity) {
            order->status = OrderStatus::FILLED;
        } else {
            order->status = OrderStatus::PARTIALLY_FILLED;
        }
        
        fills_.emplace_back(order->order_id, fill_price, fill_qty);
    }
    
    bool match_buy_order(std::shared_ptr<Order> order) {
        if (ask_book_.empty()) return false;
        
        // For FOK, check if we can fill completely
        if (order->tif == TimeInForce::FOK) {
            int64_t available_qty = 0;
            for (const auto& [price, level] : ask_book_) {
                if (price > order->price) break;
                available_qty += level->total_quantity;
            }
            
            if (available_qty < order->quantity) {
                order->status = OrderStatus::REJECTED;
                return false;
            }
        }
        
        // Process matching
        std::vector<std::shared_ptr<Order>> orders_to_remove;
        
        for (auto& [price, level] : ask_book_) {
            if (price > order->price) break;
            
            while (!level->orders.empty() && order->remaining_quantity() > 0) {
                auto resting_order = level->orders.front();
                
                if (!resting_order->is_active()) {
                    level->orders.pop();
                    continue;
                }
                
                int64_t match_qty = std::min(order->remaining_quantity(), 
                                           resting_order->remaining_quantity());
                
                process_fill(order, price, match_qty);
                process_fill(resting_order, price, match_qty);
                
                level->total_quantity -= match_qty;
                
                if (resting_order->status == OrderStatus::FILLED) {
                    level->orders.pop();
                    orders_to_remove.push_back(resting_order);
                }
                
                if (order->status == OrderStatus::FILLED) break;
            }
            
            if (level->orders.empty()) {
                orders_to_remove.push_back(nullptr);
            }
        }
        
        // Clean up
        for (auto& order_to_remove : orders_to_remove) {
            if (order_to_remove) {
                orders_.erase(order_to_remove->order_id);
            }
        }
        
        for (auto it = ask_book_.begin(); it != ask_book_.end();) {
            if (it->second->orders.empty()) {
                it = ask_book_.erase(it);
            } else {
                ++it;
            }
        }
        
        return order->is_active();
    }
    
    bool match_sell_order(std::shared_ptr<Order> order) {
        if (bid_book_.empty()) return false;
        
        // For FOK, check if we can fill completely
        if (order->tif == TimeInForce::FOK) {
            int64_t available_qty = 0;
            for (const auto& [price, level] : bid_book_) {
                if (price < order->price) break;
                available_qty += level->total_quantity;
            }
            
            if (available_qty < order->quantity) {
                order->status = OrderStatus::REJECTED;
                return false;
            }
        }
        
        // Process matching
        std::vector<std::shared_ptr<Order>> orders_to_remove;
        
        for (auto& [price, level] : bid_book_) {
            if (price < order->price) break;
            
            while (!level->orders.empty() && order->remaining_quantity() > 0) {
                auto resting_order = level->orders.front();
                
                if (!resting_order->is_active()) {
                    level->orders.pop();
                    continue;
                }
                
                int64_t match_qty = std::min(order->remaining_quantity(), 
                                           resting_order->remaining_quantity());
                
                process_fill(order, price, match_qty);
                process_fill(resting_order, price, match_qty);
                
                level->total_quantity -= match_qty;
                
                if (resting_order->status == OrderStatus::FILLED) {
                    level->orders.pop();
                    orders_to_remove.push_back(resting_order);
                }
                
                if (order->status == OrderStatus::FILLED) break;
            }
            
            if (level->orders.empty()) {
                orders_to_remove.push_back(nullptr);
            }
        }
        
        // Clean up
        for (auto& order_to_remove : orders_to_remove) {
            if (order_to_remove) {
                orders_.erase(order_to_remove->order_id);
            }
        }
        
        for (auto it = bid_book_.begin(); it != bid_book_.end();) {
            if (it->second->orders.empty()) {
                it = bid_book_.erase(it);
            } else {
                ++it;
            }
        }
        
        return order->is_active();
    }

public:
    MatchEngine() = default;
    
    // Submit a new order
    bool submit_order(const std::string& order_id, int8_t side, double price, 
                     int64_t quantity, int8_t tif) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        OrderSide order_side = static_cast<OrderSide>(side);
        TimeInForce time_in_force = static_cast<TimeInForce>(tif);
        int64_t timestamp = order_counter_.fetch_add(1);
        
        auto order = std::make_shared<Order>(order_id, order_side, price, quantity, 
                                           time_in_force, timestamp);
        
        // Handle market orders
        if (price == 0.0) {
            double best_price = 0.0;
            if (order_side == OrderSide::BUY && !ask_book_.empty()) {
                best_price = ask_book_.begin()->first;
            } else if (order_side == OrderSide::SELL && !bid_book_.empty()) {
                best_price = bid_book_.begin()->first;
            }
            
            if (best_price == 0.0) {
                order->status = OrderStatus::REJECTED;
                return false;
            }
            order->price = best_price;
        }
        
        // Try to match immediately
        if (order_side == OrderSide::BUY) {
            match_buy_order(order);
        } else {
            match_sell_order(order);
        }
        
        // Add to book if not fully matched and not IOC/FOK
        if (order->is_active() && time_in_force != TimeInForce::IOC && time_in_force != TimeInForce::FOK) {
            if (order_side == OrderSide::BUY) {
                add_order_to_bid_book(order);
            } else {
                add_order_to_ask_book(order);
            }
        } else if (order->status == OrderStatus::PENDING) {
            order->status = OrderStatus::REJECTED;
        }
        
        return order->status != OrderStatus::REJECTED;
    }
    
    // Cancel an existing order
    bool cancel_order(const std::string& order_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = orders_.find(order_id);
        if (it == orders_.end() || !it->second->is_active()) {
            return false;
        }
        
        it->second->status = OrderStatus::CANCELLED;
        
        if (it->second->side == OrderSide::BUY) {
            remove_order_from_bid_book(order_id);
        } else {
            remove_order_from_ask_book(order_id);
        }
        
        return true;
    }
    
    // Process a market tick (for testing)
    void process_tick(double /* price */, int64_t size, int8_t side) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        OrderSide order_side = static_cast<OrderSide>(side);
        std::string tick_order_id = "TICK_" + std::to_string(order_counter_.fetch_add(1));
        
        // Create a market order for the tick
        auto order = std::make_shared<Order>(tick_order_id, order_side, 0.0, size, 
                                           TimeInForce::IOC, order_counter_.fetch_add(1));
        
        if (order_side == OrderSide::BUY) {
            match_buy_order(order);
        } else {
            match_sell_order(order);
        }
    }
    
    // Get all fills since last call
    std::vector<std::tuple<std::string, double, int64_t>> get_fills() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::vector<std::tuple<std::string, double, int64_t>> result;
        result.reserve(fills_.size());
        
        for (const auto& fill : fills_) {
            result.emplace_back(fill.order_id, fill.price, fill.quantity);
        }
        
        fills_.clear();
        return result;
    }
    
    // Get order book snapshot (for debugging)
    std::map<double, int64_t> get_best_levels() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::map<double, int64_t> result;
        
        if (!bid_book_.empty()) {
            result[bid_book_.begin()->first] = bid_book_.begin()->second->total_quantity;
        }
        
        if (!ask_book_.empty()) {
            result[ask_book_.begin()->first] = ask_book_.begin()->second->total_quantity;
        }
        
        return result;
    }
    
    // Get order count
    size_t get_order_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return orders_.size();
    }
};

} // namespace flashback