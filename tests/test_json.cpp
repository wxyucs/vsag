#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

TEST_CASE("Json usage", "[json]") {
    nlohmann::json j{};

    j["name"] = "Alice";
    j["age"] = 25;
    j["isMarried"] = false;

    std::string name = j["name"];
    int age = j["age"];
    bool isMarried = j["isMarried"];

    REQUIRE(name == "Alice");
    REQUIRE(age == 25);
    REQUIRE(isMarried == false);
}
