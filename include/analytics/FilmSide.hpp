#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace video_engine {

enum class FilmSide {
    FrontView,
    FrontLeftView,
    FrontRightView,
    LeftSideView,
    RightSideView,
    RearLeftView,
    RearView,
    RearRightView,
};

inline FilmSide parseFilmSide(std::string_view value) {
    if (value == "front_view" || value == "front") {
        return FilmSide::FrontView;
    }
    if (value == "front_left_view" || value == "front_left" ||
        value == "front left") {
        return FilmSide::FrontLeftView;
    }
    if (value == "front_right_view" || value == "front_right" ||
        value == "front right") {
        return FilmSide::FrontRightView;
    }
    if (value == "left_side_view" || value == "left") {
        return FilmSide::LeftSideView;
    }
    if (value == "right_side_view" || value == "right") {
        return FilmSide::RightSideView;
    }
    if (value == "rear_left_view" || value == "rear_left" ||
        value == "rear left") {
        return FilmSide::RearLeftView;
    }
    if (value == "rear_view" || value == "rear") {
        return FilmSide::RearView;
    }
    if (value == "rear_right_view" || value == "rear_right" ||
        value == "rear right") {
        return FilmSide::RearRightView;
    }
    throw std::invalid_argument(
        "Unsupported film side: " + std::string(value) +
        ". Use front_view, front_left_view, front_right_view, left_side_view, "
        "right_side_view, rear_left_view, rear_view, or rear_right_view.");
}

inline const char* filmSideName(FilmSide side) noexcept {
    switch (side) {
        case FilmSide::FrontView:
            return "front_view";
        case FilmSide::FrontLeftView:
            return "front_left_view";
        case FilmSide::FrontRightView:
            return "front_right_view";
        case FilmSide::LeftSideView:
            return "left_side_view";
        case FilmSide::RightSideView:
            return "right_side_view";
        case FilmSide::RearLeftView:
            return "rear_left_view";
        case FilmSide::RearView:
            return "rear_view";
        case FilmSide::RearRightView:
            return "rear_right_view";
    }
    return "front_view";
}

inline bool filmSidePrefersLeftBodyChain(FilmSide side) noexcept {
    return side == FilmSide::FrontLeftView ||
           side == FilmSide::LeftSideView ||
           side == FilmSide::RearLeftView;
}

inline bool filmSidePrefersRightBodyChain(FilmSide side) noexcept {
    return side == FilmSide::FrontRightView ||
           side == FilmSide::RightSideView ||
           side == FilmSide::RearRightView;
}

}  // namespace video_engine
