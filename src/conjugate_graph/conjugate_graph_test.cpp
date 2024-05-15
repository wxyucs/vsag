#include "conjugate_graph.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("build & add", "[ut][conjugate_graph]") {
    SECTION("build and add neighbor") {
        std::shared_ptr<vsag::ConjugateGraph> conjugate_graph =
            std::make_shared<vsag::ConjugateGraph>();
        REQUIRE(conjugate_graph->AddNeighbor(0, 0) == true);
        REQUIRE(conjugate_graph->AddNeighbor(0, 0) == false);
        REQUIRE(conjugate_graph->AddNeighbor(0, 1) == true);
        REQUIRE(conjugate_graph->AddNeighbor(1, 0) == true);
    }

    SECTION("memory usage") {
        std::shared_ptr<vsag::ConjugateGraph> conjugate_graph =
            std::make_shared<vsag::ConjugateGraph>();
        REQUIRE(conjugate_graph->GetMemoryUsage() == 4);

        conjugate_graph->AddNeighbor(0, 0);
        REQUIRE(conjugate_graph->GetMemoryUsage() == 28);

        conjugate_graph->AddNeighbor(0, 1);
        REQUIRE(conjugate_graph->GetMemoryUsage() == 36);

        conjugate_graph->AddNeighbor(1, 0);
        REQUIRE(conjugate_graph->GetMemoryUsage() == 60);
    }

    SECTION("serialize and deserialize") {
        std::shared_ptr<vsag::ConjugateGraph> conjugate_graph =
            std::make_shared<vsag::ConjugateGraph>();
        vsag::BinarySet binary_set;
        REQUIRE_THROWS(conjugate_graph->Serialize());
        REQUIRE_THROWS(conjugate_graph->Deserialize(binary_set));
    }
}
