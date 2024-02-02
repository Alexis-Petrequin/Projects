#include "graphic.hpp"

Bomb::Bomb(int power, float x, float z, char state, Character *character)
{
    this->_pos.x = x;
    this->_pos.y = 0.70f;
    this->_pos.z = z;
    this->_tcreated = clock();
    this->_state = state;
    this->_power = power;
    this->radius[0] = 1; // la range gauche de la bombe avec prise en compte des cubes
    this->radius[1] = 1; // la range haut de la bombe avec prise en compte des cubes
    this->radius[2] = 1; // la range droite de la bombe avec prise en compte des cubes
    this->radius[3] = 1; // la range bas de la bombe avec prise en compte des cubes
    this->character = character;
}

Bomb::Bomb(Character *character, int radius0, int radius1, int radius2, int radius3, float x, float z, int power, char state, int tcreated)
{
    this->character = character;
    this->radius[0] = radius0; // la range gauche de la bombe avec prise en compte des cubes
    this->radius[1] = radius1; // la range haut de la bombe avec prise en compte des cubes
    this->radius[2] = radius2; // la range droite de la bombe avec prise en compte des cubes
    this->radius[3] = radius3; // la range bas de la bombe avec prise en compte des cubes
    this->_pos.x = x;
    this->_pos.y = 0.70f;
    this->_pos.z = z;
    this->_power = power;
    this->_state = state;
    this->_tcreated = tcreated;
}

void Game::drop_bomb(Character *charact)  //pose une bombe lorsque le joueur appuie sur la touche pour poser une bombe
{
    float x = (int)(charact->character_pos.x + 0.50f) + 0.00f;    // permet de centrer la bombe sur la case
    float z = (int)(charact->character_pos.z + 0.50f) + 0.00f;    // permet de centrer la bombe sur la case
    std::vector<Bomb>::iterator it;

    if (IsKeyReleased(charact->_key_bomb) && charact->_nbbomb > 0) {
        for (it = _bomb.begin(); it != _bomb.end(); ++it)
            if ((it->_state == 'a' && it->_pos.x == x && it->_pos.z == z) || //check si il y a déjà une bombe à l'emplacement du character
            charact->_live < 1 || //check si le character est mort
            (charact->_obstacle == false && in_block(&_map, charact) == true)) //check si le character est dans une caisse grace a un bonus
                return;
        Bomb new_bomb(charact->_power, x, z, 'a', charact);
        _bomb.push_back(new_bomb);
        charact->_nbbomb--;
    }
}