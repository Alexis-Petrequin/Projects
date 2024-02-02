#include "graphic.hpp"

Game::Game()
{
    _texture = new Texture2D[32];
    _texture[0] = LoadTexture("ressources/ground.png");
    _texture[1] = LoadTexture("ressources/uwall.png");
    _texture[2] = LoadTexture("ressources/scenery.png");
    _texture[3] = LoadTexture("ressources/box.png");
    _texture[4] = LoadTexture("ressources/wall.png");
    _texture[5] = LoadTexture("ressources/fire.png");
    _texture[6] = LoadTexture("ressources/bonus1.png");
    _texture[7] = LoadTexture("ressources/bonus2.png");
    _texture[8] = LoadTexture("ressources/bonus3.png");
    _texture[9] = LoadTexture("ressources/bonus4.png");
    _texture[10] = LoadTexture("ressources/gameover.png");
    _texture[11] = LoadTexture("ressources/restart.png");
    _texture[12] = LoadTexture("ressources/exit.png");
    _texture[13] = LoadTexture("ressources/resume.png");
    _texture[14] = LoadTexture("ressources/save.png");
    _texture[15] = LoadTexture("ressources/menu/25.png");
    _texture[16] = LoadTexture("ressources/menu/50.png");
    _texture[17] = LoadTexture("ressources/menu/75.png");
    _texture[18] = LoadTexture("ressources/menu/100.png");
    _texture[19] = LoadTexture("ressources/menu/back.png");
    _texture[20] = LoadTexture("ressources/menu/exit.png");
    _texture[21] = LoadTexture("ressources/menu/music.png");
    _texture[22] = LoadTexture("ressources/menu/mute.png");
    _texture[23] = LoadTexture("ressources/menu/new.png");
    _texture[24] = LoadTexture("ressources/menu/options.png");
    _texture[25] = LoadTexture("ressources/menu/play.png");
    _texture[26] = LoadTexture("ressources/menu/save.png");
    _texture[27] = LoadTexture("ressources/menu/sound.png");
    _texture[28] = LoadTexture("ressources/menu1.png");
    _texture[29] = LoadTexture("ressources/menu2.png");
    _texture[30] = LoadTexture("ressources/menu/plus.png");
    _texture[31] = LoadTexture("ressources/menu/less.png");
    _sound = new Sound[5];
    _sound[0] = LoadSound("ressources/expl.ogg");
    _sound[1] = LoadSound("ressources/bonus.wav");
    _sound[2] = LoadSound("ressources/die.wav");
    _sound[3] = LoadSound("ressources/click.wav");
    _sound[4] = LoadSound("ressources/theme.wav");
    for (int i = 0; i < 5; i ++)
        SetSoundVolume( _sound[i], 0.80f);
    _font = LoadFontEx("ressources/font.ttf", 64, 0, 250);
    _character = new Character[4];
    _character[0]._key_down = KEY_DOWN;
    _character[0]._key_up = KEY_UP;
    _character[0]._key_left = KEY_LEFT;
    _character[0]._key_right = KEY_RIGHT;
    _character[0]._key_bomb = KEY_SPACE;
    _character[0].character_pos = { 0.00f, 0.5f, 0.00f};
    _character[1].character_pos = { 12.00f, 0.5f, 0.00f};
    _character[2].character_pos = { 0.00f, 0.5f, 10.00f};
    _character[3].character_pos = { 12.00f, 0.5f, 10.00f};
    _camera = { { 6.12f, 22.50f, 15.38f }, { 6.12f, -6.29f, 2.62f }, { 0.0f, 1.0f, 0.0f }, 30.00f, CAMERA_PERSPECTIVE };
    _mode = 1;
    _nb_ia = 1;
    _nb_player = 1;
    _time = 0;
    _bonus.push_back(Bonus(-10.00f, -10.00f, 'r'));  //evite les segv
    _bomb.push_back(Bomb(_character->_power, -10.00f, -10.00f, 'c', _character));  //evite peut etre les segv/invalid read
}

Game::~Game()
{
    for (int i = 0; i < 32; i++)
        UnloadTexture(_texture[i]);
    UnloadFont(_font);
    delete _texture;
    StopSoundMulti();
    for (int i = 0; i < 5; i++)
        UnloadSound(_sound[i]);
    delete _sound;
}

void Game::restart()
{
    _mode = 1;
    _character[0].character_pos = { 0.00f, 0.5f, 0.00f};
    _character[1].character_pos = { 12.00f, 0.5f, 0.00f};
    _character[2].character_pos = { 0.00f, 0.5f, 10.00f};
    _character[3].character_pos = { 12.00f, 0.5f, 10.00f};
    for (int i = 0; i < 4; i++) {
        _character[i]._live = 1;
        _character[i]._nbbomb = 1;
        _character[i]._obstacle = true;
        _character[i]._power = 2;
        _character[i]._speed = 4.00f;
        _character[i]._tobstacle = 0;
        _character[i]._tspeed = 0;
        _character[i]._Ia_goal = { -1.00f, -1.00f };
    }
    for (int i = _nb_player; i < _nb_player + _nb_ia; i++) {
        _character[i]._key_up = 0;
        _character[i]._key_down = 0;
        _character[i]._key_left = 0;
        _character[i]._key_right = 0;
        _character[i]._key_bomb = 0;
    }
    std::vector<Bomb>::iterator it;
    for (it = _bomb.begin(); it != _bomb.end(); ++it) {
        if (it->_state != 'c') {
            _bomb.erase(it);
            --it;
        }
    }
    std::vector<Bonus>::iterator it2;
    for (it2 = _bonus.begin(); it2 != _bonus.end(); ++it2) {
        if (it2->_type != 'r') {
            _bonus.erase(it2);
            --it2;
        }
    }
}

void Game::init_game(int nb_player, int nb_ia, std::map<Vector2, char> level)
{
    _map = level;        // charge la map
    add_border(&_map);   // ajouter les bords a la map
    _nb_ia = nb_ia;
    _nb_player = nb_player;
    if (_nb_player == 2) {
        _character[1]._key_down = KEY_S;
        _character[1]._key_up = KEY_W;
        _character[1]._key_left = KEY_A;
        _character[1]._key_right = KEY_D;
        _character[1]._key_bomb = KEY_E;
    }
}

void Game::init_loop()
{
    UpdateCamera(&_camera);
    BeginDrawing();
    ClearBackground(BLACK);
}

void Game::collision()
{
    fire_col_bomb(); // collision entre une explosion et une bombe et la fait exploser
    fire_col_player(); // collision entre une explosion et un joueur et le tue
    fire_col_bonus(); //collision entre une explosion et un bonus et le detruit
    player_col_bonus(); // collision entre le joueur et un bonus et l'active
}

int Game::play(int nb_player, int nb_ia, std::map<Vector2, char> level)
{
    init_game(nb_player, nb_ia, level);

    while (!WindowShouldClose() && _mode)
    {
        init_loop();
        collision();
        if (!IsSoundPlaying(_sound[4]))
            PlaySound(_sound[4]);
        if (_mode == 1) {
            for (int i = 0; i < _nb_player; i++) {
                move(&_character[i], &_map, &_bomb);
                drop_bomb(&_character[i]);
            }
            ia();
            timeout_bonus();
        }
        BeginMode3D(_camera);
        draw_3D();
        EndMode3D();
        draw2D();
        EndDrawing();
        if (_mode > 3)
            return _mode - 2;
    }
    return (0);
}