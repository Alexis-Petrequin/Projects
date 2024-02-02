/*
** EPITECH PROJECT, 2019
** map
** File description:
** about map
*/

#include "../../include/mylist.h"

int aggro(map_t *map)
{
    int aggro = rand() % 80;
    sfVector2f perso = sfSprite_getPosition(map->perso->perso);

    if ((perso.y > 11315 || perso.x > 12660) && !aggro)
        return (1);
    return (0);
}

int draw_game_over(map_t *m, player_t *player)
{
    sfVector2f cam = sfView_getCenter(m->camera);
    sfVector2f pos[2] = { {cam.x, cam.y - 100}, {cam.x, cam.y} };

    if (player->monsterkilled == 10 && player->quest == 1) {
        player->state = 2;
        for (int i = 5; i < 7; i++) {
            sfRectangleShape_setPosition(m->interface[i]->sprite, pos[i - 5]);
            sfText_setPosition(m->interface[i]->text, pos[i - 5]);
        }
    }
}

static void start_map(map_t *map, player_t *player)
{
    sfVector2f reset = { 8480, 6455 };

    if (player->state == 2) {
        player->monsterkilled = 0;
        player->level = 1;
        player->nbpotfire = 1;
        player->nbpothp = 1;
        player->nbpotmana = 1;
        player->quest = 0;
        sfSprite_setPosition(map->perso->perso, reset);
    }
    draw_statut(map, player, 0);
    draw_inventory(map, player, 0);
    draw_quit(map, 0, player);
    sfMusic_play(map->music);
    sfMusic_setLoop(map->music, sfTrue);
    sfMusic_setVolume(map->music, player->mvol);
    player->state = 4;
}

int play_map(sfRenderWindow *window, map_t *map, player_t *player)
{
    int end = 0;

    start_map(map, player);
    while (!end) {
        if (move_map(window, map, player))
            end = 3;
        draw_map(window, map, player);
        while (sfRenderWindow_pollEvent(window, map->event))
            end = event_map(window, map, player);
    }
    sfView_setCenter(map->camera, (sfVector2f){960, 513});
    sfRenderWindow_setView(window, map->camera);
    sfMusic_stop(map->music);
    return (end);
}