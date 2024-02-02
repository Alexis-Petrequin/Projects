/*
** EPITECH PROJECT, 2019
** map_button
** File description:
** about map_button
*/

#include "../../include/mylist.h"

int on_button_map(sfVector2i mouse_pos, button_t *button, sfView *cam)
{
    sfVector2f view = sfView_getCenter(cam);
    sfVector2f sprite_pos = sfRectangleShape_getPosition(button->sprite);
    sfFloatRect dim = sfRectangleShape_getGlobalBounds(button->sprite);
    sfVector2f mouse = { mouse_pos.x, mouse_pos.y };

    mouse.x += 1920 * ((view.x - 960) / 1920);
    mouse.y += 1026 * ((view.y - 513) / 1026);
    if (mouse.x >= sprite_pos.x - dim.width / 2 &&
    mouse.x <= sprite_pos.x + dim.width / 2 &&
    mouse.y >= sprite_pos.y - dim.height / 2 &&
    mouse.y <= sprite_pos.y + dim.height / 2)
        return (1);
    return (0);
}

static void int_but_map2(map_t *m, int i)
{
    sfVector2f scale = { 1.1, 1.1 };
    sfVector2f unscale = { 1, 1 };
    sfColor c = sfBlue;

    sfRectangleShape_setScale(m->interface[i]->sprite, unscale);
    sfText_setScale(m->interface[i]->text, unscale);
    if (sfRectangleShape_getFillColor(m->interface[i]->sprite).b == c.b)
        sfText_setColor(m->interface[i]->text, sfYellow);
}

void int_but_map(sfVector2i mouse, map_t *m)
{
    sfVector2f scale = { 1.1, 1.1 };
    sfVector2f unscale = { 1, 1 };
    sfColor c = sfBlue;

    for (int i = 0; i < 7; i++)
        if (on_button_map(mouse, m->interface[i], m->camera)) {
            sfRectangleShape_setScale(m->interface[i]->sprite, scale);
            sfText_setScale(m->interface[i]->text, scale);
            if (m->event->type == sfEvtMouseButtonReleased)
                sfSound_play(m->sound);
            if (m->event->type == sfEvtMouseButtonReleased &&
            sfRectangleShape_getFillColor(m->interface[i]->sprite).b == c.b)
                sfText_setColor(m->interface[i]->text, sfYellow);
            if (m->event->type == sfEvtMouseButtonPressed)
                sfText_setColor(m->interface[i]->text, sfBlue);
        } else
            int_but_map2(m, i);
}