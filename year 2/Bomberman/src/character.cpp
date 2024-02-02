#include "graphic.hpp"

Character::Character()
{
    model = LoadModel("ressources/model.iqm");
    texture = LoadTexture("ressources/player_texture.png");
    SetMaterialTexture(&model.materials[0], MAP_DIFFUSE, texture);
    animsCount = 0;
    anims = LoadModelAnimations("ressources/Macarena_Animation.iqm", &animsCount);
    animFrameCounter = 0;
    _key_up = 0;
    _key_down = 0;
    _key_left = 0;
    _key_right = 0;
    _key_bomb = 0;
    _live = 1;          // determine si le character est en vie (vie = 1) ou mort (vie = 0)
    _nbbomb = 1;        // determine le nombre de bombe que peut poser le character
    _power = 2;         // determine la range des bombe du character
    _speed = 4.0f;      // determine la vitesse du character
    _obstacle = true;   // true = collision avec les obstacles activé
    character_pos = {0, 0, 0};
    _tspeed = clock();  // utilisé pour stoper le bonus de speed
    _tobstacle = clock();   // utilisé pour stopper le bonus obstacle
    _Ia_goal = { -1.00f, -1.00f };
}

Character::~Character()
{
    UnloadTexture(texture);
    for (int i = 0; i < animsCount; i++) UnloadModelAnimation(anims[i]);
    RL_FREE(anims);
    UnloadModel(model);
}

bool player_col_bomb_left(Character *character, std::vector<Bomb> *bomb)
{
    std::vector<Bomb>::iterator it = bomb->begin();

    for (it = bomb->begin(); it != bomb->end(); ++it)
        if (it->_state == 'a')
            if (character->character_pos.x - 0.61f > it->_pos.x - 0.10f && character->character_pos.x - 0.61f < it->_pos.x + 0.10f && character->character_pos.z > it->_pos.z - 0.50f && character->character_pos.z < it->_pos.z + 0.50f)
                return (true);
    return (false);
}

bool player_col_bomb_right(Character *character, std::vector<Bomb> *bomb)
{
    std::vector<Bomb>::iterator it = bomb->begin();

    for (it = bomb->begin(); it != bomb->end(); ++it)
        if (it->_state == 'a')
            if (character->character_pos.x + 0.61f > it->_pos.x - 0.10f && character->character_pos.x + 0.61f < it->_pos.x + 0.10f && character->character_pos.z > it->_pos.z - 0.50f && character->character_pos.z < it->_pos.z + 0.50f)
                return (true);
    return (false);
}

bool player_col_bomb_up(Character *character, std::vector<Bomb> *bomb)
{
    std::vector<Bomb>::iterator it = bomb->begin();

    for (it = bomb->begin(); it != bomb->end(); ++it)
        if (it->_state == 'a')
            if (character->character_pos.x > it->_pos.x - 0.50f && character->character_pos.x < it->_pos.x + 0.50f && character->character_pos.z - 0.61f > it->_pos.z - 0.10f &&  character->character_pos.z - 0.61f < it->_pos.z + 0.10f)
                return (true);
    return (false);
}

bool player_col_bomb_down(Character *character, std::vector<Bomb> *bomb)
{
    std::vector<Bomb>::iterator it = bomb->begin();

    for (it = bomb->begin(); it != bomb->end(); ++it)
        if (it->_state == 'a')
            if (character->character_pos.x > it->_pos.x - 0.50f && character->character_pos.x < it->_pos.x + 0.50f && character->character_pos.z + 0.61f > it->_pos.z - 0.10f &&  character->character_pos.z + 0.61f < it->_pos.z + 0.10f)
                return (true);
    return (false);
}

bool player_col_cube_right(Character *character, std::map<Vector2, char> *map)
{
    std::map<Vector2,char>::iterator it = map->begin();

    for (it = map->begin(); it != map->end(); ++it) {
        if ((it->second == '*' || (it->second == '&' && character->_obstacle == true) || it->second == 'b'))
            if (character->character_pos.x + 0.20f > it->first.x - 0.50f && character->character_pos.x + 0.20f < it->first.x + 0.50f &&
            character->character_pos.z < it->first.y + 0.50f && character->character_pos.z > it->first.y - 0.50f)
                return (true);
        }
    return (false);
}

bool player_col_cube_left(Character *character, std::map<Vector2, char> *map)
{
    std::map<Vector2,char>::iterator it = map->begin();

    for (it = map->begin(); it != map->end(); ++it) {
        if ((it->second == '*' || (it->second == '&' && character->_obstacle == true) || it->second == 'b'))
            if (character->character_pos.x - 0.20f > it->first.x - 0.50f && character->character_pos.x - 0.20f < it->first.x + 0.50f &&
            character->character_pos.z < it->first.y + 0.50f && character->character_pos.z > it->first.y - 0.50f)
                return (true);
        }
    return (false);
}

bool player_col_cube_up(Character *character, std::map<Vector2, char> *map)
{
    std::map<Vector2,char>::iterator it = map->begin();

    for (it = map->begin(); it != map->end(); ++it) {
        if ((it->second == '*' || (it->second == '&' && character->_obstacle == true) || it->second == 'b'))
            if (character->character_pos.x > it->first.x - 0.50f && character->character_pos.x < it->first.x + 0.50f &&
            character->character_pos.z - 0.20f < it->first.y + 0.50f && character->character_pos.z - 0.20f > it->first.y - 0.50f)
                return (true);
        }
    return (false);
}

bool player_col_cube_down(Character *character, std::map<Vector2, char> *map)
{
    std::map<Vector2,char>::iterator it = map->begin();

    for (it = map->begin(); it != map->end(); ++it) {
        if ((it->second == '*' || (it->second == '&' && character->_obstacle == true) || it->second == 'b'))
            if (character->character_pos.x > it->first.x - 0.50f && character->character_pos.x < it->first.x + 0.50f &&
            character->character_pos.z + 0.20f < it->first.y + 0.50f && character->character_pos.z + 0.20f > it->first.y - 0.50f)
                return (true);
        }
    return (false);
}

int animate(Character *character, int way, std::map<Vector2, char> *map, std::vector<Bomb> *bomb)
{
    UpdateModelAnimation(character->model, character->anims[0], character->animFrameCounter++);
    if (character->animFrameCounter >= character->anims[0].frameCount) //reset l'animation si elle est fini
        character->animFrameCounter = 0;
    if (way == 1 && player_col_cube_right(character, map) == false && player_col_bomb_right(character, bomb) == false) // permet au joueur d'avancer si il n'y à pas de cube ni de bombes dans sa direction
        character->character_pos.x += GetFrameTime() * character->_speed;
    if (way == 2 && player_col_cube_left(character, map) == false && player_col_bomb_left(character, bomb) == false)
        character->character_pos.x -= GetFrameTime() * character->_speed;
    if (way == 3 && player_col_cube_up(character, map) == false && player_col_bomb_up(character, bomb) == false)
        character->character_pos.z -= GetFrameTime() * character->_speed;
    if (way == 4 && player_col_cube_down(character, map) == false && player_col_bomb_down(character, bomb) == false)
        character->character_pos.z += GetFrameTime() * character->_speed;
    return (0);
}

int move(Character *character, std::map<Vector2, char> *map, std::vector<Bomb> *bomb)
{
    if (IsKeyDown(character->_key_right)) {
        character->model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 180.0f }); //rotate le skin
        return (animate(character, 1, map, bomb));
    }
    if (IsKeyDown(character->_key_left)) {
        character->model.transform = MatrixRotateXYZ({0.0f, 0.0f, 45.0f });
        return (animate(character, 2, map, bomb));
    }
    if (IsKeyDown(character->_key_up)) {
        character->model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 135.0f });
        return (animate(character, 3, map, bomb));
    }
    if (IsKeyDown(character->_key_down)) {
        character->model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 270.0f });
        return (animate(character, 4, map, bomb));
    }
    character->animFrameCounter = 0;    // met l'animation a 0 si le perso est immobile
    UpdateModelAnimation(character->model, character->anims[0], character->animFrameCounter);
    return (0);
}
