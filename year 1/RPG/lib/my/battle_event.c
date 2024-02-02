/*
** EPITECH PROJECT, 2020
** battle_event
** File description:
** about battle event
*/

#include "../../include/mylist.h"

void int_but_battle(sfVector2i mouse, sfRenderWindow *window, battle_t *m)
{
    sfVector2f scale = { 1.1, 1.1 };
    sfVector2f unscale = { 1, 1 };

    for (int i = 1; i < 13; i++)
        if (on_button(mouse, m->button[i])) {
            sfRectangleShape_setScale(m->button[i]->sprite, scale);
            sfText_setScale(m->button[i]->text, scale);
            if (m->event->type == sfEvtMouseButtonReleased)
                sfSound_play(m->sound);
            if (m->event->type == sfEvtMouseButtonReleased &&
            sfRectangleShape_getFillColor(m->button[i]->sprite).b == sfBlue.b)
                sfText_setColor(m->button[i]->text, sfYellow);
            if (m->event->type == sfEvtMouseButtonPressed)
                sfText_setColor(m->button[i]->text, sfBlue);
        } else {
            sfRectangleShape_setScale(m->button[i]->sprite, unscale);
            sfText_setScale(m->button[i]->text, unscale);
            if (sfRectangleShape_getFillColor(m->button[i]->sprite).b ==
            sfBlue.b)
                sfText_setColor(m->button[i]->text, sfYellow);
        }
}

void int_but_battle2(sfVector2i mouse, sfRenderWindow *window, battle_t *m)
{
    sfVector2f scale = { 1.1, 1.1 };
    sfVector2f unscale = { 1, 1 };

    for (int i = 19; i < 27; i++)
        if (on_button(mouse, m->button[i])) {
            sfRectangleShape_setScale(m->button[i]->sprite, scale);
            sfText_setScale(m->button[i]->text, scale);
            if (m->event->type == sfEvtMouseButtonReleased)
                sfSound_play(m->sound);
            if (m->event->type == sfEvtMouseButtonReleased &&
            sfRectangleShape_getFillColor(m->button[i]->sprite).b == sfBlue.b)
                sfText_setColor(m->button[i]->text, sfYellow);
            if (m->event->type == sfEvtMouseButtonPressed)
                sfText_setColor(m->button[i]->text, sfBlue);
        } else {
            sfRectangleShape_setScale(m->button[i]->sprite, unscale);
            sfText_setScale(m->button[i]->text, unscale);
            if (sfRectangleShape_getFillColor(m->button[i]->sprite).b ==
            sfBlue.b)
                sfText_setColor(m->button[i]->text, sfYellow);
        }
}

static int event_battle3(battle_t *menu, sfVector2i mouse, player_t *player)
{
    if (on_button(mouse, menu->button[9]))
        return (pothp_battle(menu, player));
    if (on_button(mouse, menu->button[10]))
        return (potmana_battle(menu, player));
    if (on_button(mouse, menu->button[11]))
        return (potfire_battle(menu, player));
    if (on_button(mouse, menu->button[12]))
        return (move_button_bat(menu->button, -2000));
    if (on_button(mouse, menu->button[20])) {
        player->state = 2;
        return (2);
    }
    if (on_button(mouse, menu->button[23])) {
        lvl_up(player);
        return (4);
    }
    if (on_button(mouse, menu->button[25]))
        return (2);
    if (on_button(mouse, menu->button[26]))
        return (draw_leave(menu, 0));
    return (0);
}

static int event_battle2(battle_t *menu, sfVector2i mouse, player_t *player)
{
    if (on_button(mouse, menu->button[1]))
        return (move_button_bat(menu->button, -2000));
    if (on_button(mouse, menu->button[2]))
        return (defend_battle(menu, player));
    if (on_button(mouse, menu->button[3]))
        return (move_button_bat(menu->button, 2000));
    if (on_button(mouse, menu->button[4]))
        return (4);
    if (on_button(mouse, menu->button[5]))
        return (cut_battle(menu, player));
    if (on_button(mouse, menu->button[6]))
        return (magic_battle(menu, player));
    if (on_button(mouse, menu->button[7]))
        return (heal_battle(menu, player));
    if (on_button(mouse, menu->button[8]))
        return (move_button_bat(menu->button, 2000));
    return (event_battle3(menu, mouse, player));
}

int event_battle(battle_t *menu, sfRenderWindow *window, player_t *player)
{
    sfVector2i mouse = sfMouse_getPositionRenderWindow(window);

    int_but_battle(mouse, window, menu);
    int_but_battle2(mouse, window, menu);
    if (menu->event->type == sfEvtClosed)
        return (1);
    if (menu->event->type == sfEvtKeyReleased &&
    menu->event->key.code == sfKeyEscape)
        if (sfRectangleShape_getPosition(menu->button[24]->sprite).x != 960 &&
            menu->mob[1]->pv > 0 && player->pv > 0)
            return (draw_leave(menu, 1));
        else
            return (draw_leave(menu, 0));
    if (menu->event->type == sfEvtMouseButtonReleased) {
        return (event_battle2(menu, mouse, player));
    }
    return (0);
}