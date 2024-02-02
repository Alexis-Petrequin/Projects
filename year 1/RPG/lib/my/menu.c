/*
** EPITECH PROJECT, 2020
** menu
** File description:
** about menu
*/

#include "../../include/mylist.h"

int gest_vol(menu_t *m, player_t *pl, int nb)
{
    char str[4] = { '\0' };

    if (nb == 1 && pl->mvol <= 95) {
        sfMusic_setVolume(m->ress->music, pl->mvol += 5);
        sfText_setString(m->button[7]->text, musicstr(pl->mvol, str));
    }
    if (nb == 2 && pl->mvol >= 5) {
        sfMusic_setVolume(m->ress->music, pl->mvol -= 5);
        sfText_setString(m->button[7]->text, musicstr(pl->mvol, str));
    }
    if (nb == 3 && pl->svol <= 95) {
        sfSound_setVolume(m->sound, pl->svol += 5);
        sfText_setString(m->button[10]->text, musicstr(pl->svol, str));
    }
    if (nb == 4 && pl->svol >= 5) {
        sfSound_setVolume(m->sound, pl->svol -= 5);
        sfText_setString(m->button[10]->text, musicstr(pl->svol, str));
    }
    return (0);
}

static int gest_event_menu2(menu_t *m, player_t *player, sfVector2i mouse)
{
    if (on_button(mouse, m->button[0]) || on_button(mouse, m->button[11]))
        return (move_button(m->button, -3000, 0));
    if (on_button(mouse, m->button[1]) || on_button(mouse, m->button[14]))
        return (move_button(m->button, 3000, 0));
    if (on_button(mouse, m->button[5]))
        return (gest_vol(m, player, 1));
    if (on_button(mouse, m->button[4]))
        return (gest_vol(m, player, 2));
    if (on_button(mouse, m->button[9]))
        return (gest_vol(m, player, 3));
    if (on_button(mouse, m->button[8]))
        return (gest_vol(m, player, 4));
    if (on_button(mouse, m->button[2]))
        return (1);
    if (on_button(mouse, m->button[12])) {
        player->state = 2;
        return (4);
    }
    return (0);
}

int gest_event_menu(sfRenderWindow *window, menu_t *m, player_t *player)
{
    sfVector2i mouse = sfMouse_getPositionRenderWindow(window);

    interaction_button(mouse, window, m);
    if (m->event->type == sfEvtClosed)
        return (1);
    if (m->event->type == sfEvtMouseButtonReleased &&
    m->event->mouseButton.button == sfMouseLeft) {
        if (on_button(mouse, m->button[13]) && player->state > 2 &&
        player->state < 5)
            return (player->state);
        return (gest_event_menu2(m, player, mouse));
    }
    return (0);
}

int draw_all_menu(sfRenderWindow *window, menu_t *m)
{
    sfVector2f pos = { 1920, 0 };
    sfTime time = sfClock_getElapsedTime(m->clock);

    for (int i = 0; m->sprite[i] != NULL; i++) {
        if (sfSprite_getPosition(m->sprite[i]->sprite).x <= -1920)
            sfSprite_setPosition(m->sprite[i]->sprite, pos);
        if (time.microseconds / 1000000.0 > 0.01)
            sfSprite_move(m->sprite[i]->sprite, m->sprite[i]->speed);
        sfRenderWindow_drawSprite(window, m->sprite[i]->sprite, NULL);
    }
    for (int i = 0; m->button[i] != NULL; i++) {
        sfRenderWindow_drawRectangleShape(window, m->button[i]->sprite, NULL);
        sfRenderWindow_drawText(window, m->button[i]->text, NULL);
    }
    if (time.microseconds / 1000000.0 > 0.01)
        sfClock_restart(m->clock);
}

int play_menu(menu_t*menu, player_t *player, sfRenderWindow *window)
{
    int end = 0;

    sfMusic_play(menu->ress->music);
    while (!end) {
        sfRenderWindow_display(window);
        sfRenderWindow_clear(window, sfBlack);
        draw_all_menu(window, menu);
        while (sfRenderWindow_pollEvent(window, menu->event) && !end)
            end = gest_event_menu(window, menu, player);
    }
    sfMusic_stop(menu->ress->music);
    return (end);
}