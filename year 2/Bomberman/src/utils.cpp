#include "graphic.hpp"

bool operator<(const Vector2 &a, const Vector2 &b) {
    if (a.y < b.y)
        return true;
    else if (a.y == b.y && a.x < b.x)
        return true;
    return false;
}

std::string get_arg(std::string str, int nb)
{
    std::string data;
    int temp = 0;
    int y = 0;
    int i = 0;

    for (i = 0; i < str.length(); i++) {
        if (temp == nb && str[i] != ';') {
            data.push_back(str[i]) ;
            y++;
        }
        if (str[i] == ';')
            temp++;
    }
    if (temp < nb)
        data = "NULL";
    return (data);
}

Character *Game::find_char(float x, float z)
{
    for (int i = 0; i < 4; i++)
        if (x == _character[i].character_pos.x && z == _character[i].character_pos.z )
            return (&_character[i]);
    return (&_character[0]);
}

int Game::load()
{
    std::ifstream file("save.txt");
    std::string buff;
    int player = 0;

    if(file.fail())
        return (-1);

    while (std::getline(file, buff)) {
        if (buff.compare(std::string("map;")) == 0)
            break;
        if (buff.compare(std::string("player 0;")) == 0) {
            player = 0;
            continue;
        }
        if (buff.compare(std::string("player 1;")) == 0) {
            player = 1;
            continue;
        }
        if (buff.compare(std::string("player 2;")) == 0) {
            player = 2;
            continue;
        }
        if (buff.compare(std::string("player 3;")) == 0) {
            player = 3;
            continue;
        }
        _character[player].character_pos.x = std::stof(get_arg(buff, 0));
        _character[player].character_pos.y = std::stof(get_arg(buff, 1));
        _character[player].character_pos.z = std::stof(get_arg(buff, 2));
        _character[player]._live = std::stoi(get_arg(buff, 3));
        _character[player]._nbbomb = std::stoi(get_arg(buff, 4));
        _character[player]._power = std::stoi(get_arg(buff, 5));
        _character[player]._speed = std::stof(get_arg(buff, 6));
        _character[player]._obstacle = std::stoi(get_arg(buff, 7));
        _character[player]._tspeed = clock() - std::stoi(get_arg(buff, 8));
        _character[player]._tobstacle = clock() - std::stoi(get_arg(buff, 9));
        _character[player]._Ia_goal.x = std::stof(get_arg(buff, 10));
        _character[player]._Ia_goal.y = std::stof(get_arg(buff, 11));
        _character[player]._key_up = std::stoi(get_arg(buff, 12));
        _character[player]._key_down = std::stoi(get_arg(buff, 13));
        _character[player]._key_left = std::stoi(get_arg(buff, 14));
        _character[player]._key_right = std::stoi(get_arg(buff, 15));
        _character[player]._key_bomb = std::stoi(get_arg(buff, 16));
    }
    std::getline(file, buff);
    _map.clear();
    for (int i = 0; i < buff.length(); i++) {
        if (get_arg(buff, i * 3 + 2) == "NULL")
            break;
        _map.insert(std::pair<Vector2, char>({ std::stof(get_arg(buff, i * 3)), std::stof(get_arg(buff, i * 3 + 1)) }, get_arg(buff, i * 3 + 2)[0]));
    }
    _bonus.clear();
    while (std::getline(file, buff)) {
        if (buff.compare(std::string("bomb;")) == 0)
            break;
        if (buff.compare(std::string("bonus;")) == 0)
            continue;
        Bonus temp(std::stof(get_arg(buff, 0)), std::stof(get_arg(buff, 2)), get_arg(buff, 3)[0], std::stoi(get_arg(buff, 5)), std::stof(get_arg(buff, 4)));
        _bonus.push_back(temp);
    }
    _bomb.clear();
    while (std::getline(file, buff)) {
        if (buff.compare(std::string("other;")) == 0)
            break;
        if (buff.compare(std::string("bomb;")) == 0)
            continue;
        Bomb bmb(find_char(std::stof(get_arg(buff, 0)), std::stof(get_arg(buff, 1))), std::stoi(get_arg(buff, 2)), std::stoi(get_arg(buff, 3)), std::stoi(get_arg(buff, 4)), std::stoi(get_arg(buff, 5)), std::stof(get_arg(buff, 6)), std::stof(get_arg(buff, 8)), std::stoi(get_arg(buff, 9)), get_arg(buff, 10)[0], clock() - std::stoi(get_arg(buff, 11)));
        _bomb.push_back(bmb);
    }
    std::getline(file, buff);
    _time = clock() + std::stoi(get_arg(buff, 0));
    std::getline(file, buff);
    _mode = std::stoi(get_arg(buff, 0));
    std::getline(file, buff);
    _nb_player = std::stoi(get_arg(buff, 0));
    std::getline(file, buff);
    _nb_ia = std::stoi(get_arg(buff, 0));
    file.close();
    return (0);
}

int Game::save()
{
    remove("save.txt");
    std::ofstream file("save.txt");

    for (int i = 0; i < _nb_ia + _nb_player; i++) {
        file << "player " << i << ";" << std::endl;
        file << _character[i].character_pos.x << ";";
        file << _character[i].character_pos.y << ";";
        file << _character[i].character_pos.z << ";";
        file << _character[i]._live << ";";
        file << _character[i]._nbbomb << ";";
        file << _character[i]._power << ";";
        file << _character[i]._speed << ";";
        file << _character[i]._obstacle << ";";
        file << _character[i]._tspeed << ";";
        file << _character[i]._tobstacle << ";";
        file << _character[i]._Ia_goal.x << ";";
        file << _character[i]._Ia_goal.y << ";";
        file << _character[i]._key_up << ";";
        file << _character[i]._key_down << ";";
        file << _character[i]._key_left << ";";
        file << _character[i]._key_right << ";";
        file << _character[i]._key_bomb << ";" << std::endl;
    }

    std::map<Vector2,char>::iterator it = _map.begin();
    file << "map;" << std::endl;
    for (it = _map.begin(); it != _map.end(); ++it) {
        file << it->first.x << ";";
        file << it->first.y << ";";
        file << it->second << ";";
    }
    file << std::endl;

    std::vector<Bonus>::iterator it2 = _bonus.begin();
    file << "bonus;" << std::endl;
    for (it2 = _bonus.begin(); it2 != _bonus.end(); ++it2) {
        file << it2->_pos.x << ";";
        file << it2->_pos.y << ";";
        file << it2->_pos.z << ";";
        file << it2->_type << ";";
        file << it2->_up << ";";
        file << ((clock() - it2->_tcreated) * 1000 / CLOCKS_PER_SEC) << ";";
        file << std::endl;
    }

    std::vector<Bomb>::iterator it3;
    file << "bomb;" << std::endl;
    for (it3 = _bomb.begin(); it3 != _bomb.end(); ++it3) {
        file << it3->character->character_pos.x << ";";
        file << it3->character->character_pos.z << ";";
        file << it3->radius[0] << ";";
        file << it3->radius[1] << ";";
        file << it3->radius[2] << ";";
        file << it3->radius[3] << ";";
        file << it3->_pos.x << ";";
        file << it3->_pos.y << ";";
        file << it3->_pos.z << ";";
        file << it3->_power << ";";
        file << it3->_state << ";";
        file << ((clock() - it3->_tcreated) * 1000 / CLOCKS_PER_SEC) << ";";
        file << std::endl;
    }
    file << "other;" << std::endl;
    file << ((clock() - _time) * 1000 / CLOCKS_PER_SEC) << std::endl;
    file << _mode << std::endl;
    file << _nb_player << std::endl;
    file << _nb_ia << std::endl;
    file.close();
    return (0);
}

void camerapos(Camera *camera) //temporaire pour les tests
{
    char buff[200] = { 0 };

    SetCameraMode(*camera, CAMERA_FREE);
    snprintf(buff, 200, "{%.2fl;%.2fl;%.2fl} {%.2fl;%.2fl;%.2fl} {%.2fl;%.2fl;%.2fl} %.2fl", camera->position.x, camera->position.y, camera->position.z, camera->target.x, camera->target.y, camera->target.z, camera->up.x, camera->up.y, camera->up.z, camera->fovy);
    DrawText(buff, 10, 10, 30, RED);
}