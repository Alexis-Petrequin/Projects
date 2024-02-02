/*
** EPITECH PROJECT, 2020
** battle
** File description:
** about battle
*/

#include "../../include/mylist.h"

void update_mob(battle_t *m, player_t *player)
{
    int b = rand() % 3;

    sfSprite_setTexture(m->mob[1]->sprite, m->ress->texture[b + 12], sfFalse);
    m->mob[1]->pvmax = ((b + 1) * 20) + (player->level * 10);
    m->mob[1]->manamax = ((b + 1) * 15) + (player->level * 5);
    m->mob[1]->force = ((b + 1) * 5) + (player->level * 2.5);
    m->mob[1]->defense = ((b + 1) * 1.5) + (player->level * 1);
    m->mob[1]->pv = m->mob[1]->pvmax;
    m->mob[1]->mana = m->mob[1]->manamax;
}

static void start_battle2(battle_t *m, player_t *player, sfColor original)
{
    draw_leave(m, 0);
    if (sfRectangleShape_getPosition(m->button[0]->sprite).y != 910)
        mvbt_battle(m->button, 1200, 0);
    if (sfRectangleShape_getPosition(m->button[0]->sprite).y == 2910)
        mvbt_battle(m->button, -2000, 0);
    if (sfRectangleShape_getPosition(m->button[1]->sprite).x == 3470)
        move_button_bat(m->button, -2000);
    if (sfRectangleShape_getPosition(m->button[1]->sprite).x == -530)
        move_button_bat(m->button, 2000);
}

static void start_battle(battle_t *m, player_t *player, sfColor original)
{
    sfVector2f reset = { -1000, -1000 };

    sfSprite_setColor(m->mob[1]->sprite, original);
    sfSound_setVolume(m->sound, player->svol);
    sfMusic_setVolume(m->ress->music, player->mvol);
    sfMusic_play(m->ress->music);
    sfMusic_setLoop(m->ress->music, sfTrue);
    if (player->state == 4) {
        update_mob(m, player);
        update_player(player);
    }
    for (int i = 21; i <= 23; i++) {
        sfRectangleShape_setPosition(m->button[i]->sprite, reset);
        sfText_setPosition(m->button[i]->text, reset);
    }
    start_battle2(m, player, original);
    player->state = 3;
}

int play_battle(battle_t *menu, sfRenderWindow *window, player_t *player)
{
    int end = 0;
    sfUint8 buffer[400 * 300 * 4] = { 0 };
    sfColor original = sfSprite_getColor(menu->mob[1]->sprite);
    sfColor original2 = sfSprite_getColor(menu->mob[0]->sprite);

    original.a = 255;
    start_battle(menu, player, original);
    while (!end) {
        sfRenderWindow_display(window);
        sfRenderWindow_clear(window, sfBlack);
        update_particles(buffer, menu);
        draw_dmg(menu);
        draw_b(window, menu, player, buffer);
        mob_turn(menu, player, original);
        draw_over(menu, player);
        draw_win(menu, player);
        while (sfRenderWindow_pollEvent(window, menu->event))
            end = event_battle(menu, window, player);
    }
    sfMusic_stop(menu->ress->music);
    return (end);
}