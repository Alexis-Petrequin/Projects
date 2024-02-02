#include "graphic.hpp"

void Game::draw2D()
{
    draw_ath();
    draw_pause();
    draw_game_over();
}

void Game::draw_game_over()
{
    int count = 0;
    Vector2 mouse = GetMousePosition();
    char buff[20] = { 0 };

    for (int i = 0; i < _nb_player + _nb_ia; i++)
        if (_character[i]._live > 0) {
            snprintf(buff, 20, "PLAYER %d WIN!!", i + 1);
            count++;
        }
    if (count > 1)
        return;
    _mode = 3;
    if (count == 1)
        DrawTextEx(_font, buff, { 650.00f, 700.00f}, 120.00f, 0, WHITE);
    DrawTextureEx(_texture[10], {575.00f, -100.00f}, 0.00f, 1.50f, WHITE);
    if (mouse.x > 580.00f && mouse.x < 890 && mouse.y > 450 && mouse.y < 576 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[11], {580.00f, 450.00f}, 0.00f, 0.40f, GRAY);
    else if (mouse.x > 580.00f && mouse.x < 890 && mouse.y > 450 && mouse.y < 576)
        DrawTextureEx(_texture[11], {580.00f, 450.00f}, 0.00f, 0.40f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[11], {580.00f, 450.00f}, 0.00f, 0.40f, WHITE);
    if (mouse.x > 1030.00f && mouse.x < 1340.00f && mouse.y > 450.00f && mouse.y < 576 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[12], {1030.00f, 450.00f}, 0.00f, 0.40f, GRAY);
    else if (mouse.x > 1030.00f && mouse.x < 1340.00f && mouse.y > 450.00f && mouse.y < 576)
        DrawTextureEx(_texture[12], {1030.00f, 450.00f}, 0.00f, 0.40f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[12], {1030.00f, 450.00f}, 0.00f, 0.40f, WHITE);
    if (mouse.x > 580.00f && mouse.x < 890 && mouse.y > 450 && mouse.y < 576 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        _mode = 4; // restart
    }
    if (mouse.x > 1030.00f && mouse.x < 1340.00f && mouse.y > 450.00f && mouse.y < 576 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        _mode = 5; //exit main menu
        PlaySoundMulti(_sound[3]);
    }
}

void Game::draw_pause()
{
    if (_mode > 2)
        return;
    Vector2 mouse = GetMousePosition();
    if (_mode == 2 && IsKeyReleased(KEY_ESCAPE)) {
        std::vector<Bomb>::iterator it;
        for (it = _bomb.begin(); it != _bomb.end(); ++it)
            it->_tcreated += clock() - (clock_t)_time;
        for (int i = 0; i < _nb_player + _nb_ia; i++) {
            _character[i]._tobstacle += clock() - _time;
            _character[i]._tspeed += clock() - _time;
        }
        _mode = 1;
    }
    else if (_mode == 1 && IsKeyReleased(KEY_ESCAPE)) {
        _time = clock();
        _mode = 2;
    }
    if (_mode == 1)
        return;
    if (mouse.x > 800.00f && mouse.x < 1130.00f && mouse.y > 450.00f && mouse.y < 576 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[12], {800.00f, 450.00f}, 0.00f, 0.40f, GRAY);
    else if (mouse.x > 800.00f && mouse.x < 1130.00f && mouse.y > 450.00f && mouse.y < 576)
        DrawTextureEx(_texture[12], {800.00f, 450.00f}, 0.00f, 0.40f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[12], {800.00f, 450.00f}, 0.00f, 0.40f, WHITE);
    if (mouse.x > 895.00f && mouse.x < 1020 && mouse.y > 600 && mouse.y < 723 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        DrawTextureEx(_texture[14], {895.00f, 600.00f}, 0.00f, 0.40f, GRAY);
    else if (mouse.x > 895.00f && mouse.x < 1020 && mouse.y > 600 && mouse.y < 723)
        DrawTextureEx(_texture[14], {895.00f, 600.00f}, 0.00f, 0.40f, LIGHTGRAY);
    else
        DrawTextureEx(_texture[14], {895.00f, 600.00f}, 0.00f, 0.40f, WHITE);
    if (mouse.x > 800.00f && mouse.x < 1130.00f && mouse.y > 450.00f && mouse.y < 576 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        _mode = 5;
    }
    if (mouse.x > 895.00f && mouse.x < 1020 && mouse.y > 600 && mouse.y < 723 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        PlaySoundMulti(_sound[3]);
        save();
    }
}

void Game::draw_ath()
{
    char buff[15] = { 0 };

    DrawTextEx(_font, "PLAYER 1", {15.00f, 8.00f}, 45.00f, 1, YELLOW);
    DrawTextureEx(_texture[9], {50.00f, 95.00f}, 180.00f, 0.90f, WHITE);
    DrawTextureEx(_texture[8], {50.00f, 145.00f}, 180.00f, 0.96f, WHITE);
    snprintf(buff, 15, "%d", _character[0]._nbbomb);
    DrawTextEx(_font, buff, { 65.00f, 58.00f }, 45.00, 0, YELLOW);
    snprintf(buff, 15, "%d", _character[0]._power - 1);
    DrawTextEx(_font, buff, { 65.00f, 109.00f }, 45.00, 0, YELLOW);
    if (_character[0]._speed == 6.00f) {
        DrawTextureEx(_texture[7], {50.00f, 192.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 5 - ((clock() - _character[0]._tspeed ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 5 - ((_time - _character[0]._tspeed ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 65.00f, 157.00f }, 45.00, 0, YELLOW);
    }
    if (_character[0]._obstacle == false) {
        DrawTextureEx(_texture[6], {50.00f, 240.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 3 - ((clock() - _character[0]._tobstacle ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 3 - ((_time - _character[0]._tobstacle ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 65.00f, 205.00f }, 45.00, 0, YELLOW);
    }
    if (_nb_ia + _nb_player == 1)
        return;
    DrawTextEx(_font, "PLAYER 2", {1750.00f, 8.00f}, 45.00f, 1, YELLOW);
    DrawTextureEx(_texture[9], {1905.00f, 95.00f}, 180.00f, 0.90f, WHITE);
    DrawTextureEx(_texture[8], {1905.00f, 145.00f}, 180.00f, 0.96f, WHITE);
    snprintf(buff, 15, "%d", _character[1]._nbbomb);
    DrawTextEx(_font, buff, { 1837.00f, 58.00f }, 45.00, 0, YELLOW);
    snprintf(buff, 15, "%d", _character[1]._power - 1);
    DrawTextEx(_font, buff, { 1837.00f, 109.00f }, 45.00, 0, YELLOW);
    if (_character[1]._speed == 6.00f) {
        DrawTextureEx(_texture[7], {1905.00f, 192.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 5 - ((clock() - _character[1]._tspeed ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 5 - ((_time - _character[1]._tspeed ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 1820.00f, 157.00f }, 45.00, 0, YELLOW);
    }
    if (_character[1]._obstacle == false) {
        DrawTextureEx(_texture[6], {1905.00f, 240.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 3 - ((clock() - _character[1]._tobstacle ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 3 - ((_time - _character[1]._tobstacle ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 1820.00f, 205.00f }, 45.00, 0, YELLOW);
    }
    if (_nb_ia + _nb_player == 2)
        return;
    DrawTextEx(_font, "PLAYER 3", {15.00f, 970.00f}, 45.00f, 1, YELLOW);
    DrawTextureEx(_texture[9], {50.00f, 960.00f}, 180.00f, 0.90f, WHITE);
    DrawTextureEx(_texture[8], {50.00f, 912.00f}, 180.00f, 0.96f, WHITE);
    snprintf(buff, 15, "%d", _character[2]._nbbomb);
    DrawTextEx(_font, buff, { 65.00f, 925.00f }, 45.00, 0, YELLOW);
    snprintf(buff, 15, "%d", _character[2]._power - 1);
    DrawTextEx(_font, buff, { 65.00f, 875.00f }, 45.00, 0, YELLOW);
    if (_character[2]._speed == 6.00f) {
        DrawTextureEx(_texture[7], {50.00f, 864.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 5 - ((clock() - _character[2]._tspeed ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 5 - ((_time - _character[2]._tspeed ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 65.00f, 830.00f }, 45.00, 0, YELLOW);
    }
    if (_character[2]._obstacle == false) {
        DrawTextureEx(_texture[6], {50.00f, 816.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 3 - ((clock() - _character[2]._tobstacle ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 3 - ((_time - _character[2]._tobstacle ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 65.00f, 780.00f }, 45.00, 0, YELLOW);
    }

    if (_nb_ia + _nb_player == 3)
        return;
    DrawTextEx(_font, "PLAYER 4", {1750.00f, 970.00f}, 45.00f, 1, YELLOW);
    DrawTextureEx(_texture[9], {1905.00f, 960.00f}, 180.00f, 0.90f, WHITE);
    DrawTextureEx(_texture[8], {1905.00f, 912.00f}, 180.00f, 0.96f, WHITE);
    snprintf(buff, 15, "%d", _character[3]._nbbomb);
    DrawTextEx(_font, buff, { 1837.00f, 925.00f }, 45.00, 0, YELLOW);
    snprintf(buff, 15, "%d", _character[3]._power - 1);
    DrawTextEx(_font, buff, { 1837.00f, 875.00f }, 45.00, 0, YELLOW);
    if (_character[3]._speed == 6.00f) {
        DrawTextureEx(_texture[7], {1905.00f, 864.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 5 - ((clock() - _character[3]._tspeed ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 5 - ((_time - _character[3]._tspeed ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 1820.00f, 830.00f }, 45.00, 0, YELLOW);
    }
    if (_character[3]._obstacle == false) {
        DrawTextureEx(_texture[6], {1905.00f, 816.00f}, 180.00f, 0.86f, WHITE);
        if (_mode == 1)
            snprintf(buff, 15, "%llds", 3 - ((clock() - _character[3]._tobstacle ) / CLOCKS_PER_SEC));
        else
            snprintf(buff, 15, "%llds", 3 - ((_time - _character[3]._tobstacle ) / CLOCKS_PER_SEC));
        DrawTextEx(_font, buff, { 1820.00f, 780.00f }, 45.00, 0, YELLOW);
    }
}