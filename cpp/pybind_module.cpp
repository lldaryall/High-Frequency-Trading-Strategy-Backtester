#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "match_engine.cpp"

namespace py = pybind11;

PYBIND11_MODULE(_match_cpp, m) {
    m.doc() = "High-performance C++ matching engine for Flashback HFT backtester";
    
    // Enums
    py::enum_<flashback::OrderSide>(m, "OrderSide")
        .value("BUY", flashback::OrderSide::BUY)
        .value("SELL", flashback::OrderSide::SELL);
    
    py::enum_<flashback::TimeInForce>(m, "TimeInForce")
        .value("DAY", flashback::TimeInForce::DAY)
        .value("IOC", flashback::TimeInForce::IOC)
        .value("FOK", flashback::TimeInForce::FOK);
    
    py::enum_<flashback::OrderStatus>(m, "OrderStatus")
        .value("PENDING", flashback::OrderStatus::PENDING)
        .value("PARTIALLY_FILLED", flashback::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", flashback::OrderStatus::FILLED)
        .value("CANCELLED", flashback::OrderStatus::CANCELLED)
        .value("REJECTED", flashback::OrderStatus::REJECTED);
    
    // MatchEngine class
    py::class_<flashback::MatchEngine>(m, "MatchEngine")
        .def(py::init<>())
        .def("submit_order", &flashback::MatchEngine::submit_order,
             "Submit a new order to the matching engine",
             py::arg("order_id"), py::arg("side"), py::arg("price"), 
             py::arg("quantity"), py::arg("tif"))
        .def("cancel_order", &flashback::MatchEngine::cancel_order,
             "Cancel an existing order",
             py::arg("order_id"))
        .def("process_tick", &flashback::MatchEngine::process_tick,
             "Process a market tick (for testing)",
             py::arg("price"), py::arg("size"), py::arg("side"))
        .def("get_fills", &flashback::MatchEngine::get_fills,
             "Get all fills since last call",
             py::return_value_policy::move)
        .def("get_best_levels", &flashback::MatchEngine::get_best_levels,
             "Get best bid/ask levels (for debugging)",
             py::return_value_policy::move)
        .def("get_order_count", &flashback::MatchEngine::get_order_count,
             "Get current number of active orders");
    
    // Fill tuple type
    m.def("create_fill", [](const std::string& order_id, double price, int64_t quantity) {
        return std::make_tuple(order_id, price, quantity);
    }, "Create a fill tuple", py::arg("order_id"), py::arg("price"), py::arg("quantity"));
}
