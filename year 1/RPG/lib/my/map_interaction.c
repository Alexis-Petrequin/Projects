/*
** EPITECH PROJECT, 2019
** map_interaction
** File description:
** about interaction of map
*/

#include "../../include/mylist.h"

int draw_bubble2(map_t *map, sfRenderWindow *window, int *next, int i)
{
    if (map->event->key.code == 'e') {
        next[i] += 1;
        map->event->key.code = 0;
    }
    if (next[i] > 3)
        next[i] = 0;
    if (next[i] > 0) {
        sfRenderWindow_drawRectangleShape(window,
        map->bubble[i][next[i] - 1].sprite, NULL);
        sfRenderWindow_drawText(window,
        map->bubble[i][next[i] - 1].text, NULL);
    }
    return (next[i]);
}

int draw_bubble(map_t *map, sfRenderWindow *window, player_t *player)
{
    sfVector2f posp = sfSprite_getPosition(map->perso->perso);
    sfVector2f poss = { 0, 0 };
    static int next[3] = { 0, 0, 0 };

    for (int i = 0; map->pnj[i] != NULL; i++) {
        poss = sfSprite_getPosition(map->pnj[i]->sprite);
        if (poss.x + 50 > posp.x && poss.x - 50 < posp.x &&
            poss.y + 50 > posp.y && poss.y - 50 < posp.y) {
            next[i] = draw_bubble2(map, window, next, i);
        }
        else
            next[i] = 0;
    }
    if (next[2] == 3 && player->quest == 0)
        player->quest = 1;
    for (int i = 0; i < 3; i++)
        if (next[i] > 0) {
            map->tuto = 1;
            return (0);
        }
    map->tuto = 0;
}

void pnj_info(map_t *map, sfRenderWindow *window)
{
    sfVector2f pospl = sfSprite_getPosition(map->perso->perso);
    sfVector2f pospnj = { 0, 0 };

    for (int i = 0; map->pnj[i] != NULL; ++i) {
        pospnj = sfSprite_getPosition(map->pnj[i]->sprite);
        if (pospnj.x + 50 > pospl.x && pospnj.x - 50 < pospl.x &&
            pospnj.y + 50 > pospl.y && pospnj.y - 50 < pospl.y ) {
            pospnj.x += 10;
            pospnj.y -= 10;
            sfText_setPosition(map->presse, pospnj);
            sfRenderWindow_drawText(window, map->presse, NULL);
        }
    }
}