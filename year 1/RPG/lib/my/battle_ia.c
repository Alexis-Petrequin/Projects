/*
** EPITECH PROJECT, 2020
** battle_ia
** File description:
** about ia
*/

#include "../../include/mylist.h"

void mob_attack2(battle_t *menu, player_t *player, int dmg, sfVector2f pos)
{
    char str[6] = { '\0' };

    if (dmg > 0) {
        player->pv -= dmg;
        sfText_setColor(menu->button[18]->text, sfRed);
        sfText_setString(menu->button[18]->text, numtostr(dmg, str));
        sfText_setPosition(menu->button[18]->text, pos);
        menu->mob[0]->rect.top = 1089;
        menu->mob[0]->rect.left = 0;
    } else {
        sfText_setColor(menu->button[18]->text, sfYellow);
        sfText_setString(menu->button[18]->text, "BLOCKED !");
        sfText_setPosition(menu->button[18]->text, pos);
        if (player->defense > 1000)
            player->defense -= 1000;
    }
}

int mob_attack(battle_t *menu, player_t *player, sfColor original)
{
    sfTime time = sfClock_getElapsedTime(menu->clock[3]);
    int dmg = rand() % menu->mob[1]->force + (menu->mob[1]->force / 2);
    sfVector2f pos = { 860, 720 };

    dmg -= player->defense;
    draw_leave(menu, 0);
    sfSprite_setColor(menu->mob[1]->sprite, sfRed);
    if (time.microseconds / 1000000.0 > 4.00) {
        mob_attack2(menu, player, dmg, pos);
        mvbt_battle(menu->button, -800, 0);
        sfSprite_setColor(menu->mob[1]->sprite, original);
        return (0);
    }
    return (2);
}

int mob_heal(battle_t *menu, player_t *player, sfColor original)
{
    sfTime time = sfClock_getElapsedTime(menu->clock[3]);
    int heal = menu->mob[1]->pvmax * 0.3;
    char str[6] = { '\0' };

    draw_leave(menu, 0);
    sfSprite_setColor(menu->mob[1]->sprite, sfGreen);
    if (time.microseconds / 1000000.0 > 4.00) {
        menu->mob[1]->pv += heal;
        sfText_setColor(menu->button[17]->text, sfGreen);
        sfText_setString(menu->button[17]->text, numtostr(heal, str));
        sfText_setPosition(menu->button[17]->text, (sfVector2f){ 450, 690 });
        mvbt_battle(menu->button, -800, 0);
        sfSprite_setColor(menu->mob[1]->sprite, original);
        menu->mob[1]->mana -= 35;
        if (player->defense > 1000)
            player->defense -= 1000;
        return (0);
    }
    return (1);
}

int mob_turn(battle_t *m, player_t *player, sfColor original)
{
    sfTime time = sfClock_getElapsedTime(m->clock[3]);
    static int choice = 0;

    if (m->mob[1]->pv <= 0)
        return (0);
    if (time.microseconds / 1000000.0 <= 2.50)
        draw_leave(m, 0);
    if (choice == 0)
        choice = rand() % 5 + 1;
    if (sfRectangleShape_getPosition(m->button[0]->sprite).y == 1710.00 &&
    time.microseconds / 1000000.0 > 2.50)
        if (choice == 1 && m->mob[1]->pv <= m->mob[1]->pvmax * 0.4 &&
        m->mob[1]->mana >= 35)
            choice = mob_heal(m, player, original);
        else
            choice = mob_attack(m, player, original);
    return (0);
}

void draw_dmg(battle_t *menu)
{
    sfVector2f pos = sfText_getPosition(menu->button[18]->text);
    sfTime time = sfClock_getElapsedTime(menu->clock[4]);

    pos.y -= 1;
    if (pos.y > 670 && time.microseconds / 1000000.0 > 0.01) {
        sfText_setPosition(menu->button[18]->text, pos);
        sfClock_restart(menu->clock[4]);
    }
    if (pos.y <= 670)
        sfText_setColor(menu->button[18]->text, sfTransparent);
    pos = sfText_getPosition(menu->button[17]->text);
    pos.y -= 1;
    if (pos.y > 640 && time.microseconds / 1000000.0 > 0.01) {
        sfText_setPosition(menu->button[17]->text, pos);
        sfClock_restart(menu->clock[4]);
    }
    if (pos.y <= 640)
        sfText_setColor(menu->button[17]->text, sfTransparent);
}