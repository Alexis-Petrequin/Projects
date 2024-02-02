/*
** EPITECH PROJECT, 2019
** map_draw_all2
** File description:
** about map_draw_all2
*/

#include "../../include/mylist.h"

void draw_statut(map_t *map, player_t *pl, int nb)
{
    sfVector2f c = sfView_getCenter(map->camera);
    int max = 17;
    sfVector2f pos[19] = { {c.x - 30, c.y - 400}, {c.x - 350, c.y - 350},
    {c.x - 480, c.y - 290}, {c.x - 300, c.y - 290}, {c.x - 480, c.y - 230},
    {c.x - 300, c.y - 230}, {c.x - 480, c.y - 170}, {c.x - 300, c.y - 170},
    {c.x - 480, c.y - 110}, {c.x - 300, c.y - 110}, {c.x - 480, c.y - 50},
    {c.x - 300, c.y - 50}, {c.x - 50, c.y + 100}, {c.x + 80, c.y + 100},
    {c.x + 350, c.y - 350}, {c.x + 130, c.y - 290}, {c.x + 500, c.y - 290},
    {c.x - 550, c.y - 420}, {0, 0} };

    max = update_statut(map, pl);
    if (nb == 1 && pl->state == 4) {
        sfRectangleShape_setPosition(map->menu->rectangle[0], pos[17]);
        for (int i = 0; i < max; i++)
            sfText_setPosition(map->menu->text[i], pos[i]);
    }
    if (nb == 0) {
        sfRectangleShape_setPosition(map->menu->rectangle[0], pos[18]);
        for (int i = 0; i < 17; i++)
            sfText_setPosition(map->menu->text[i], pos[18]);
    }
}