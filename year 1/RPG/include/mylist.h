/*
** EPITECH PROJECT, 2019
** mylist
** File description:
** header.h
*/

#include <SFML/Graphics.h>
#include <SFML/System.h>
#include <SFML/Audio.h>
#include <SFML/Window.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct sfvector4f_s
{
    float x;
    float y;
    float a;
    float b;
} sfvector4f_t;

typedef struct ressource_s
{
    sfFont *police;
    sfTexture **texture;
    sfMusic *music;
    sfSoundBuffer *s_buffer;
} ressource_t;

typedef struct button_s
{
    sfText *text;
    sfRectangleShape *sprite;
} button_t;

typedef struct sprite_s
{
    sfSprite *sprite;
    sfVector2f pos;
    sfVector2f scale;
    sfVector2f speed;
} sprite_t;

typedef struct menu_s
{
    sfEvent *event;
    ressource_t *ress;
    button_t **button;
    sfSound *sound;
    sprite_t **sprite;
    sfClock *clock;
} menu_t;

typedef struct mob_s
{
    sfSprite *sprite;
    sfIntRect rect;
    int pvmax;
    int pv;
    int manamax;
    int mana;
    int force;
    int defense;
} mob_t;

typedef struct battle_s
{
    sfEvent *event;
    ressource_t *ress;
    button_t **button;
    sfSound *sound;
    sprite_t **sprite;
    sfClock **clock;
    mob_t **mob;
    int fire;
} battle_t;

typedef struct player_s
{
    int level;
    int pvmax;
    int pv;
    int manamax;
    int mana;
    int force;
    int defense;
    int magic;
    int state;
    float svol;
    float mvol;
    int nbpothp;
    int nbpotmana;
    int nbpotfire;
    int monsterkilled;
    int quest;
} player_t;

typedef struct perso_s
{
    sfSprite *perso;
    sfSprite *map;
    sfIntRect rect;
    sfVector2f pos;
} perso_t;

typedef struct vector4i_s
{
    int left;
    int right;
    int up;
    int down;
} vector4i_t;

typedef struct sprit_s
{
    int type;
    sfSprite *sprite;
} sprit_t;

typedef struct tab_s
{
    sfText **text;
    sfRectangleShape **rectangle;
} tab_t;

typedef struct map_s
{
    sfEvent *event;
    sfView *camera;
    perso_t *perso;
    sfTexture **texture;
    sprit_t **scenary;
    sprit_t **pnj;
    sfFont *font;
    button_t **bubble;
    sfText *presse;
    sfClock **clock;
    button_t **interface;
    sfSoundBuffer *s_buffer;
    sfSound *sound;
    tab_t *menu;
    sfMusic *music;
    int tuto;
} map_t;

int my_getnbr(char const *str);
void my_putchar(char c);
int my_putnbr(long nb);
int my_putstr(char const *str);
int my_strlen(char const *str);
sprite_t *make_sprite(sfTexture *texture, sfvector4f_t posize, sfVector2f sp);
player_t *init_player(void);
sfClock **init_clock(int nb);
sfRenderWindow *fill_mywindow(int width, int height, char const *name);
menu_t *init_menu(void);
sprite_t **init_sprite_menu(sfTexture **texture);
button_t **init_menu_butt(ressource_t *ress);
button_t *button_menu(char *text, sfvector4f_t posize, ressource_t *ress);
ressource_t *init_ress_menu(void);
sfTexture **init_textu_menu(void);
int play_menu(menu_t*menu, player_t *player, sfRenderWindow *window);
int gest_event_menu(sfRenderWindow *window, menu_t *m, player_t *player);
int draw_all_menu(sfRenderWindow *window, menu_t *m);
battle_t *init_battle(void);
mob_t **init_mob(sfTexture **texture);
mob_t *make_mob(sfIntRect rect, sfVector2f pos, sfTexture *texture);
button_t *button_battle(char *text, sfvector4f_t po, ressource_t *re, int si);
button_t **init_battle_butt(ressource_t *ress);
sprite_t **init_sprite_battle(sfTexture **text);
ressource_t *init_ress_battle(void);
sfTexture **init_textu_battle(void);
void init_textu_battle2(sfTexture **texture);
int shine_sprite(sprite_t *sprite, int nb);
void shine_all(battle_t *menu, sfTime time);
int draw_idle(sfRenderWindow *window, battle_t *m);
char *musicstr(float nb, char *str);
int gest_vol(menu_t *m, player_t *pl, int nb);
int on_button(sfVector2i mouse_pos, button_t *button);
void interaction_button(sfVector2i mouse, sfRenderWindow *window, menu_t *m);
int move_button(button_t **button, int nb, int start);
int move_button_bat(button_t **button, int nb);
void int_but_battle(sfVector2i mouse, sfRenderWindow *window, battle_t *m);
char *numtostr(int nb, char *str);
int potfire_battle(battle_t *menu, player_t *player);
int potmana_battle(battle_t *menu, player_t *player);
int pothp_battle(battle_t *menu, player_t *player);
int cut_battle(battle_t *menu, player_t *player);
int magic_battle(battle_t *menu, player_t *player);
void set_particles(sfUint8 *pixel, sfUint8 red, sfUint8 green, sfUint8 blue);
void reset_particles(sfUint8 *pixel);
void copy_particles(sfUint8 *from, sfUint8 *to);
int particles_comp(sfUint8 *pixel, sfUint8 red, sfUint8 green, sfUint8 blue);
void up_fire_part(sfUint8 *buffer, size_t i);
void mid_fire_part(sfUint8 *buffer, size_t i);
void botom_fire_part(sfUint8 *buffer, size_t i);
int update_particles(sfUint8 *buffer, battle_t *menu);
void draw_fire(sfRenderWindow *window, sfUint8 *buffer);
int mvbt_battle(button_t **button, int nb, int start);
void mob_attack2(battle_t *menu, player_t *player, int dmg, sfVector2f pos);
int mob_attack(battle_t *menu, player_t *player, sfColor original);
int mob_heal(battle_t *menu, player_t *player, sfColor original);
int mob_turn(battle_t *m, player_t *player, sfColor original);
void draw_win(battle_t *menu, player_t *player);
void draw_win2(battle_t *menu, player_t *player, int loot, int looted);
void draw_over(battle_t *menu, player_t *player);
void draw_dmg(battle_t *menu);
int heal_battle(battle_t *menu, player_t *player);
int draw_leave(battle_t *menu, int bool);
int defend_battle(battle_t *menu, player_t *player);
map_t *init_map(void);
sprit_t **init_scenary(sfTexture **texture);
int clustingcluster(sfTexture **textu, sfVector2f *dim, sprit_t **scenary);
int cluster_tree(sfTexture *t, sfVector2f dm, sfVector2f pos, sprit_t **sp);
sprit_t *init_sprite(sfTexture *t, sfVector2f pos, sfVector2f scl, int type);
button_t **fill_dial(button_t **dial, int y, int i, map_t *map);
int play_map(sfRenderWindow *window, map_t *map, player_t *player);
int move_map(sfRenderWindow *window, map_t *map, player_t *player);
int move_gestion(sfRenderWindow *window, map_t *map, vector4i_t move);
void animation(map_t *map);
void pnj_info(map_t *map, sfRenderWindow *window);
int draw_bubble(map_t *map, sfRenderWindow *window, player_t *player);
int draw_bubble2(map_t *map, sfRenderWindow *window, int *next, int i);
button_t **init_interface(map_t *map);
void draw_map(sfRenderWindow *win, map_t *map, player_t *player);
int event_map(sfRenderWindow *win, map_t *m, player_t *pl);
void int_but_map(sfVector2i mouse, map_t *m);
void draw_statut(map_t *map, player_t *pl, int nb);
int draw_inventory(map_t *map, player_t *pl, int nb);
void draw_quit(map_t *map, int nb, player_t *player);
int on_button_map(sfVector2i mouse_pos, button_t *button, sfView *cam);
int update_statut(map_t *map, player_t *pl);
void update_stuff(map_t *map, player_t *pl);
int aggro(map_t *map);
int draw_game_over(map_t *m, player_t *player);
void free_all(player_t *player, menu_t *menu, battle_t *battle, map_t *map);
void free_map2(map_t *map);
int play_battle(battle_t *menu, sfRenderWindow *window, player_t *player);
int event_battle(battle_t *menu, sfRenderWindow *window, player_t *player);
void int_but_battle2(sfVector2i mouse, sfRenderWindow *window, battle_t *m);
void update_player(player_t *player);
void update_stats(battle_t *m, player_t *player);
void lvl_up(player_t *player);
int draw_b(sfRenderWindow *win, battle_t *m, player_t *pl, sfUint8 *buff);
void update_mob(battle_t *m, player_t *player);