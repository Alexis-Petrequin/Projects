/*
** EPITECH PROJECT, 2019
** map_draw_all
** File description:
** about map_draw_all
*/

#include "../../include/mylist.h"

void draw_sprite(sfRenderWindow *win, perso_t *perso, map_t *map)
{
    sfVector2f pos = sfSprite_getPosition(perso->perso);

    for (int i = 0; map->scenary[i] != NULL; ++i) {
        if (sfSprite_getPosition(map->scenary[i]->sprite).y <= pos.y - 60)
            sfRenderWindow_drawSprite(win, map->scenary[i]->sprite, NULL);
    }
    for (int i = 0; map->pnj[i] != NULL; ++i)
        if (sfSprite_getPosition(map->pnj[i]->sprite).y <= pos.y + 25)
            sfRenderWindow_drawSprite(win, map->pnj[i]->sprite, NULL);
    sfRenderWindow_drawSprite(win, perso->perso, NULL);
    for (int i = 0; map->scenary[i] != NULL; ++i) {
        if (sfSprite_getPosition(map->scenary[i]->sprite).y > pos.y - 60)
            sfRenderWindow_drawSprite(win, map->scenary[i]->sprite, NULL);
    }
    for (int i = 0; map->pnj[i] != NULL; ++i)
        if (sfSprite_getPosition(map->pnj[i]->sprite).y > pos.y + 25)
            sfRenderWindow_drawSprite(win, map->pnj[i]->sprite, NULL);
}

void draw_button(sfRenderWindow *w, map_t *map)
{
    sfVector2f poscam = sfView_getCenter(map->camera);
    sfVector2f pos[2] = {{poscam.x - 800, poscam.y - 450},
        {poscam.x - 800, poscam.y - 350}};

    for (int i = 0; i < 7; i++) {
        if (i < 2) {
            sfRectangleShape_setPosition(map->interface[i]->sprite, pos[i]);
            sfText_setPosition(map->interface[i]->text, pos[i]);
        }
        sfRenderWindow_drawRectangleShape(w, map->interface[i]->sprite, NULL);
        sfRenderWindow_drawText(w, map->interface[i]->text, NULL);
    }
}

void draw_quit(map_t *map, int nb, player_t *player)
{
    sfVector2f poscam = sfView_getCenter(map->camera);
    sfVector2f pos[5] = {{0, 0}, {poscam.x - 800, poscam.y - 390},
    {poscam.x, poscam.y - 300}, {poscam.x - 160, poscam.y - 100},
    {poscam.x + 160, poscam.y - 100}};

    if (nb == 1 && player->state == 4)
        for (int i = 2; i < 5; i++) {
            sfRectangleShape_setPosition(map->interface[i]->sprite, pos[i]);
            sfText_setPosition(map->interface[i]->text, pos[i]);
        }
    if (nb == 0)
        for (int i = 2; i < 5; i++) {
            sfRectangleShape_setPosition(map->interface[i]->sprite, pos[0]);
            sfText_setPosition(map->interface[i]->text, pos[0]);
        }
}

int draw_inventory(map_t *map, player_t *pl, int nb)
{
    sfVector2f c = sfView_getCenter(map->camera);
    sfVector2f pos[9] = { {c.x - 30, c.y - 400}, {c.x - 480, c.y - 290},
    {c.x - 300, c.y - 290}, {c.x - 480, c.y - 230}, {c.x - 300, c.y - 230},
    {c.x - 480, c.y - 170}, {c.x - 300, c.y - 170}, {c.x - 550, c.y - 420},
    {0, 0} };

    update_stuff(map, pl);
    if (nb == 0) {
        sfRectangleShape_setPosition(map->menu->rectangle[1], pos[8]);
        for (int i = 17; i < 24; i++)
            sfText_setPosition(map->menu->text[i], pos[8]);
        return (0);
    }
    if (pl->state == 4) {
        sfRectangleShape_setPosition(map->menu->rectangle[1], pos[7]);
        for (int i = 17; i < 24; i++)
            sfText_setPosition(map->menu->text[i], pos[i - 17]);
    }
    return (0);
}

void draw_map(sfRenderWindow *win, map_t *map, player_t *player)
{
    sfRenderWindow_display(win);
    sfRenderWindow_clear(win, sfBlack);
    sfRenderWindow_drawSprite(win, map->perso->map, NULL);
    draw_sprite(win, map->perso, map);
    draw_bubble(map, win, player);
    if (sfRectangleShape_getPosition(map->menu->rectangle[0]).x > 3000)
        draw_statut(map, player, 1);
    if (sfRectangleShape_getPosition(map->menu->rectangle[1]).x > 3000)
        draw_inventory(map, player, 1);
    if (sfRectangleShape_getPosition(map->interface[2]->sprite).x > 3000)
        draw_quit(map, 1, player);
    draw_game_over(map, player);
    draw_button(win, map);
    if (map->tuto == 0)
        pnj_info(map, win);
    for (int i = 0; i < 2; i++)
        sfRenderWindow_drawRectangleShape(win, map->menu->rectangle[i], NULL);
    for (int i = 0; i < 24; i++)
        sfRenderWindow_drawText(win, map->menu->text[i], NULL);
}