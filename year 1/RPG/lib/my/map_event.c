/*
** EPITECH PROJECT, 2019
** map_event
** File description:
** about map_event
*/

#include "../../include/mylist.h"

int event_map2(sfRenderWindow *win, map_t *m, player_t *pl)
{
    sfVector2i mouse = sfMouse_getPositionRenderWindow(win);

    if (on_button_map(mouse, m->interface[4], m->camera))
        draw_quit(m, 0, pl);
    if (on_button_map(mouse, m->interface[3], m->camera))
        return (2);
    if (on_button_map(mouse, m->interface[0], m->camera))
        if (sfRectangleShape_getPosition(m->menu->rectangle[0]).x < 3000) {
            draw_quit(m, 0, pl);
            draw_inventory(m, pl, 0);
            draw_statut(m, pl, 1);
        } else
            draw_statut(m, pl, 0);
    if (on_button_map(mouse, m->interface[1], m->camera))
        if (sfRectangleShape_getPosition(m->menu->rectangle[1]).x < 3000) {
            draw_statut(m, pl, 0);
            draw_quit(m, 0, pl);
            draw_inventory(m, pl, 1);
        } else
            draw_inventory(m, pl, 0);
    return (0);
}

int event_map(sfRenderWindow *win, map_t *m, player_t *pl)
{
    sfVector2i mouse = sfMouse_getPositionRenderWindow(win);

    int_but_map(sfMouse_getPositionRenderWindow(win), m);
    if (m->event->type == sfEvtClosed)
        return (1);
    if (m->event->type == sfEvtKeyReleased &&
    m->event->key.code == sfKeyEscape)
        if (sfRectangleShape_getPosition(m->interface[2]->sprite).x < 3000) {
            draw_statut(m, pl, 0);
            draw_inventory(m, pl, 0);
            draw_quit(m, 1, pl);
        } else
            draw_quit(m, 0, pl);
    if (m->event->type == sfEvtMouseButtonReleased &&
    m->event->mouseButton.button == sfMouseLeft) {
        if (on_button_map(mouse, m->interface[6], m->camera))
            return (2);
        return (event_map2(win, m, pl));
    }
    return (0);
}
