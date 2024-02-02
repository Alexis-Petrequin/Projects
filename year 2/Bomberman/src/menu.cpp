#include "graphic.hpp"

int Game::main_menu(Vector2 mouse)
{
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 450.00f && mouse.y < 555.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[25], {1350.00f, 450.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 450.00f && mouse.y < 555.00f)
        DrawTextureEx(_texture[25], {1350.00f, 450.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[25], {1350.00f, 450.00f}, 0.00f, 2.80f, WHITE);

        if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 600.00f && mouse.y < 705.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[24], {1350.00f, 600.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 600.00f && mouse.y < 705.00f)
        DrawTextureEx(_texture[24], {1350.00f, 600.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[24], {1350.00f, 600.00f}, 0.00f, 2.80f, WHITE);

        if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[20], {1350.00f, 850.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f)
        DrawTextureEx(_texture[20], {1350.00f, 850.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[20], {1350.00f, 850.00f}, 0.00f, 2.80f, WHITE);
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 450.00f && mouse.y < 555.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (2);
    }
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 600.00f && mouse.y < 705.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (3);
    }
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (0);
    }
    return (1);
}

int Game::menu_option(Vector2 mouse)
{
    static float music = 0.80f;
    static float sound = 0.80f;

    DrawTextureEx(_texture[21], {1350.00f, 450.00f}, 0.00f, 2.80f, WHITE);
    DrawTextureEx(_texture[27], {1350.00f, 600.00f}, 0.00f, 2.80f, WHITE);
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f)
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, WHITE);

    if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 460.00f && mouse.y < 545.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[31], {1250.00f, 460.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 460.00f && mouse.y < 545.00f)
        DrawTextureEx(_texture[31], {1250.00f, 460.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[31], {1250.00f, 460.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 462.00f && mouse.y < 547.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[30], {1645.00f, 462.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 462.00f && mouse.y < 547.00f)
        DrawTextureEx(_texture[30], {1645.00f, 462.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[30], {1645.00f, 462.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 612.00f && mouse.y < 697.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[31], {1250.00f, 612.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 612.00f && mouse.y < 697.00f)
        DrawTextureEx(_texture[31], {1250.00f, 612.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[31], {1250.00f, 612.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 614.00f && mouse.y < 699.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[30], {1645.00f, 614.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 614.00f && mouse.y < 699.00f)
        DrawTextureEx(_texture[30], {1645.00f, 614.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[30], {1645.00f, 614.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 460.00f && mouse.y < 545.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (music > 0.00f) {
            music -= 0.05f;
        SetSoundVolume(_sound[4], music);
        }
    }
    if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 462.00f && mouse.y < 547.00f  && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (music < 1.00f) {
            music += 0.05f;
        SetSoundVolume(_sound[4], music);
        }
    }
    if (mouse.x > 1250.00f && mouse.x < 1335.00f && mouse.y > 612.00f && mouse.y < 697.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (sound > 0.00f) {
            sound -= 0.05f;
            for (int i = 0; i < 4; i++)
                SetSoundVolume(_sound[i], sound);
        }
    }
    if (mouse.x > 1645.00f && mouse.x < 1725.00f && mouse.y > 614.00f && mouse.y < 699.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (sound < 1.00f) {
            sound += 0.05f;
            for (int i = 0; i < 4; i++)
                SetSoundVolume(_sound[i], sound);
        }
    }

    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (1);
    }
    return (3);
}

int Game::menu_play(Vector2 mouse)
{
    char buff[5] = { 0 };

    if (_nb_ia + _nb_player > 4)
        _nb_ia = 2;
    snprintf(buff, 5, "%d", _nb_player);
    DrawTextEx(_font, "PLAYER:", {960.00f, 464.00f}, 90.00f, 1.00f, WHITE);
    DrawTextEx(_font, buff, {1460.00f, 464.00f}, 90.00f, 1.00f, WHITE);
    snprintf(buff, 5, "%d", _nb_ia);
    DrawTextEx(_font, "IA:", {960.00f, 615.00f}, 90.00f, 1.00f, WHITE);
    DrawTextEx(_font, buff, {1460.00f, 615.00f}, 90.00f, 1.00f, WHITE);

    if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 460.00f && mouse.y < 545.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[31], {1350.00f, 460.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 460.00f && mouse.y < 545.00f)
        DrawTextureEx(_texture[31], {1350.00f, 460.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[31], {1350.00f, 460.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 462.00f && mouse.y < 547.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[30], {1525.00f, 462.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 462.00f && mouse.y < 547.00f)
        DrawTextureEx(_texture[30], {1525.00f, 462.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[30], {1525.00f, 462.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 612.00f && mouse.y < 697.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[31], {1350.00f, 612.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 612.00f && mouse.y < 697.00f)
        DrawTextureEx(_texture[31], {1350.00f, 612.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[31], {1350.00f, 612.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 614.00f && mouse.y < 699.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[30], {1525.00f, 614.00f}, 0.00f, 0.25f, GRAY);
    else if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 614.00f && mouse.y < 699.00f)
        DrawTextureEx(_texture[30], {1525.00f, 614.00f}, 0.00f, 0.25f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[30], {1525.00f, 614.00f}, 0.00f, 0.25f, WHITE);

    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f)
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[19], {1350.00f, 850.00f}, 0.00f, 2.80f, WHITE);

    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 730.00f && mouse.y < 835.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[25], {1350.00f, 730.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 730.00f && mouse.y < 835.00f)
        DrawTextureEx(_texture[25], {1350.00f, 730.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[25], {1350.00f, 730.00f}, 0.00f, 2.80f, WHITE);

    if (mouse.x > 1050.00f && mouse.x < 1320.00f && mouse.y > 730.00f && mouse.y < 835.00f && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[26], {1050.00f, 730.00f}, 0.00f, 2.80f, GRAY);
    else if (mouse.x > 1050.00f && mouse.x < 1320.00f && mouse.y > 730.00f && mouse.y < 835.00f)
        DrawTextureEx(_texture[26], {1050.00f, 730.00f}, 0.00f, 2.80f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[26], {1050.00f, 730.00f}, 0.00f, 2.80f, WHITE);

    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 730.00f && mouse.y < 835.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        if ( _nb_ia + _nb_player > 1) {
            PlaySoundMulti(_sound[3]);
            return (4);
    }
    if (mouse.x > 1050.00f && mouse.x < 1320.00f && mouse.y > 730.00f && mouse.y < 835.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (5);
    }
    if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 460.00f && mouse.y < 545.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (_nb_player > 1)
            _nb_player--;
    }
    if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 462.00f && mouse.y < 547.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (_nb_player < 2)
            _nb_player++;
    }
    if (mouse.x > 1350.00f && mouse.x < 1435.00f && mouse.y > 612.00f && mouse.y < 697.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (_nb_ia > 0)
            _nb_ia--;
    }
    if (mouse.x > 1525.00f && mouse.x < 1610.00f && mouse.y > 614.00f && mouse.y < 699.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        if (_nb_player + _nb_ia < 4)
            _nb_ia++;
    }
    if (mouse.x > 1350.00f && mouse.x < 1620.00f && mouse.y > 850.00f && mouse.y < 955.00f && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        return (1);
    }
    return (2);
}

int Game::menu()
{
    int mode = 1;
    Vector2 mouse = GetMousePosition();

    while (!WindowShouldClose() && mode) {
        BeginDrawing();
        ClearBackground(BLACK);
        if (!IsSoundPlaying(_sound[4]))
            PlaySound(_sound[4]);
        mouse = GetMousePosition();
        DrawTextureEx(_texture[29], Vector2{ 0.0f, 0.0f }, 0.0, 0.65f, WHITE);
        DrawTextureEx(_texture[28], Vector2{ 400.00f, 180.0f}, 0.0, 2.30f, WHITE);

        if (mode == 1)
            mode = main_menu(mouse);
        if (mode == 2)
            mode = menu_play(mouse);
        if (mode == 3)
            mode = menu_option(mouse);
        if (mode == 4)
            return (1);
        if (mode == 5)
            return (4);
        EndDrawing();
    }
    return (0);
}