#include "raylib.h"
#include "raymath.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <iostream>
#include <string>

#ifndef GRAPHIC_HPP_
#define GRAPHIC_HPP_

#define CUBESIZE 1.00f
#define CUBEDIM CUBESIZE, CUBESIZE, CUBESIZE

class Character {
    public:
        Character();
        ~Character();
        Model model;
        Texture2D texture;
        int animsCount;
        ModelAnimation *anims;
        int animFrameCounter;
        Vector3 character_pos;
        int _live; // determine si le character est en vie (vie = 1) ou mort (vie = 0)
        int _nbbomb; // determine le nombre de bombe que peut poser le character
        int _power; // determine la range des bombe du character
        float _speed; // determine la vitesse du character
        bool _obstacle; // true = collision avec les obstacles activé
        time_t _tspeed; // utilisé pour stoper le bonus de speed
        time_t _tobstacle; // utilisé pour stopper le bonus obstacle
        Vector2 _Ia_goal;
        int _key_up;
        int _key_down;
        int _key_left;
        int _key_right;
        int _key_bomb;
};

class Bomb {
    public:
        Bomb(int power, float x, float y, char state, Character *character);
        Bomb(Character *character, int radius0, int radius1, int radius2, int radius3, float x, float z, int power, char state, int tcreated);
        ~Bomb() {};
        clock_t _tcreated; //utilisé comme décompte avant explosion ('a') et ensuite utilisé pour la durée de l'explosion ('b')
        int _power; // la range de la bombe
        Vector3 _pos;
        char _state; // 'a' signifie bombe posé au sol 'b' signifie bombe en train d'exploser
        Character *character; // le character qui a poser la bombe
        int radius[4]; // la range de la bombe avec prise en compte des cubes bloquants
};

class Bonus {
    public:
        Bonus(float x, float y, char type);
        Bonus(float x, float y, char type, int tcreated, float up);
        ~Bonus() {};
        clock_t _tcreated; //utilisé comme décompte avant explosion ('a') et ensuite utilisé pour la durée de l'explosion ('b')
        Vector3 _pos;
        char _type; // '1' = wall_pass '2' = speed up '3' = range up '4' = bomb up
        float _up;
};

class Game {
    public:
        Game();
        ~Game();
        int play(int nb_player, int nb_ia, std::map<Vector2, char> level);
        void restart();
        int menu();
        int load();
        int save();
        Camera _camera;
        Texture2D *_texture;             // liste toutes les textures pour le niveau
        Font _font;
        Sound *_sound;
        Character *_character;           // liste tout les joueurs
        std::map<Vector2, char> _map;    // liste tout les cubes de la map sauf le sol
        std::vector<Bonus> _bonus;       // liste tout les bonus sur la map
        std::vector<Bomb> _bomb;         // liste toute les bombe et explosions sur la map
        time_t _time;                     // utiliser lors de la pause
        int _mode;
        int _nb_player;
        int _nb_ia;
    private:
        void init_game(int nb_player, int nb_ia, std::map<Vector2, char> level);
        void init_loop();
        void drop_bomb(Character *charact);   //depose une bombe sur le sol lorsque le joueur le decide
        void timeout_bonus();
        void ia();
        void draw_3D();
        void draw_bomb();
        void draw_map();
        void draw_bonus();
        void draw_fire();
        void draw2D();
        void draw_ath();
        void draw_pause();
        void draw_game_over();
        void collision();
        void fire_col_bomb();
        void fire_col_player();
        void fire_col_bonus();
        void player_col_bonus();
        void fire_col_bomb_down();
        void fire_col_bomb_up();
        void fire_col_bomb_right();
        void fire_col_bomb_left();
        int radius_left(Bomb bomb);
        int radius_right(Bomb bomb);
        int radius_up(Bomb bomb);
        int radius_down(Bomb bomb);
        void fire_col_player_down();
        void fire_col_player_up();
        void fire_col_player_left();
        void fire_col_player_right();
        void fire_col_bonus_left();
        void fire_col_bonus_right();
        void fire_col_bonus_up();
        void fire_col_bonus_down();
        void draw_radius_left(Bomb bomb, char data);
        void draw_radius_right(Bomb bomb, char data);
        void draw_radius_up(Bomb bomb, char data);
        void draw_radius_down(Bomb bomb, char data);
        void IA_should_up(Character *charact);
        void IA_should_down(Character *charact);
        void IA_should_left(Character *charact);
        void IA_should_right(Character *charact);
        void IA_move();
        void Ia_find_dest(Character *IA);
        int dodge_bomb(Character *IA);
        bool still_box();
        int IA_go_box(Character *IA);
        Vector2 Ia_select_dest(Character *IA, char data);
        int dodge_right(Vector2 pos);
        int dodge_left(Vector2 pos);
        int dodge_up(Vector2 pos);
        int dodge_down(Vector2 pos);
        int IA_put_bomb(Character *IA);
        bool IA_attack(Character *IA, Character *focus);
        Vector2 Ia_find_player(Character *IA);
        int Ia_go_player(Character *IA);
        void Ia_rand_dest(Character *IA);
        int main_menu(Vector2 mouse);
        int menu_option(Vector2 mouse);
        int menu_play(Vector2 mouse);
        Character *find_char(float x, float z);
};

void add_border(std::map<Vector2, char> *map);
int move(Character *character, std::map<Vector2, char> *map, std::vector<Bomb> *bomb);
bool in_block(std::map<Vector2,char> *map, Character *character);
void camerapos(Camera *camera);     // temporaire pour les tests
bool operator<(const Vector2 &a, const Vector2 &b);
bool player_col_cube_down(Character *character, std::map<Vector2, char> *map);
bool player_col_bomb_left(Character *character, std::vector<Bomb> *bomb);
bool player_col_bomb_right(Character *character, std::vector<Bomb> *bomb);
bool player_col_bomb_up(Character *character, std::vector<Bomb> *bomb);
bool player_col_bomb_down(Character *character, std::vector<Bomb> *bomb);
bool player_col_cube_right(Character *character, std::map<Vector2, char> *map);
bool player_col_cube_left(Character *character, std::map<Vector2, char> *map);
bool player_col_cube_up(Character *character, std::map<Vector2, char> *map);
bool player_col_cube_down(Character *character, std::map<Vector2, char> *map);
std::map<Vector2, char> createMap();
std::string get_arg(std::string str, int nb);
#endif