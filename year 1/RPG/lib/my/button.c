/*
** EPITECH PROJECT, 2020
** button
** File description:
** about fonction button
*/

#include "../../include/mylist.h"

int on_button(sfVector2i mouse_pos, button_t *button)
{
    sfVector2f sprite_pos = { 0 };
    sfFloatRect dim = { 0 };

    sprite_pos = sfRectangleShape_getPosition(button->sprite);
    dim = sfRectangleShape_getGlobalBounds(button->sprite);
    if (mouse_pos.x >= sprite_pos.x - dim.width / 2 &&
    mouse_pos.x <= sprite_pos.x + dim.width / 2 &&
    mouse_pos.y >= sprite_pos.y - dim.height / 2 &&
    mouse_pos.y <= sprite_pos.y + dim.height / 2)
        return (1);
    return (0);
}

void interaction_button(sfVector2i mouse, sfRenderWindow *window, menu_t *m)
{
    sfVector2f scale = { 1.1, 1.1 };
    sfVector2f unscale = { 1, 1 };

    for (int i = 0; m->button[i] != NULL; i++)
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

int move_button(button_t **button, int nb, int start)
{
    sfVector2f pos = { 0 };

    for (int i = start; button[i] != NULL; i++) {
        pos = sfRectangleShape_getPosition(button[i]->sprite);
        pos.x += nb;
        sfRectangleShape_setPosition(button[i]->sprite, pos);
        pos = sfText_getPosition(button[i]->text);
        pos.x += nb;
        sfText_setPosition(button[i]->text, pos);
    }
        return (0);
}

int move_button_bat(button_t **button, int nb)
{
    sfVector2f pos = { 0 };

    for (int i = 1; i < 13; i++) {
        pos = sfRectangleShape_getPosition(button[i]->sprite);
        pos.x += nb;
        sfRectangleShape_setPosition(button[i]->sprite, pos);
        pos = sfText_getPosition(button[i]->text);
        pos.x += nb;
        sfText_setPosition(button[i]->text, pos);
    }
        return (0);
}