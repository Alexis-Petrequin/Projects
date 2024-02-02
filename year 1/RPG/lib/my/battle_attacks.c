/*
** EPITECH PROJECT, 2020
** battle_attacks
** File description:
** about attacks
*/

#include "../../include/mylist.h"

int mvbt_battle(button_t **button, int nb, int start)
{
    sfVector2f pos = { 0 };

    for (int i = start; i < 13; i++) {
        pos = sfRectangleShape_getPosition(button[i]->sprite);
        pos.y += nb;
        sfRectangleShape_setPosition(button[i]->sprite, pos);
        pos = sfText_getPosition(button[i]->text);
        pos.y += nb;
        sfText_setPosition(button[i]->text, pos);
    }
        return (0);
}

int cut_battle(battle_t *menu, player_t *pl)
{
    int dmg = rand() % pl->force + (pl->force / 2) - menu->mob[1]->defense;
    char str[6] = { '\0' };

    if (dmg > 0) {
        menu->mob[1]->pv -= dmg;
        sfText_setColor(menu->button[17]->text, sfRed);
        sfText_setString(menu->button[17]->text, numtostr(dmg, str));
        sfText_setPosition(menu->button[17]->text, (sfVector2f){ 450, 690 });
    } else {
        sfText_setColor(menu->button[17]->text, sfYellow);
        sfText_setString(menu->button[17]->text, "BLOCKED !");
        sfText_setPosition(menu->button[17]->text, (sfVector2f){ 450, 690 });
    }
    menu->mob[0]->rect.top = 0;
    menu->mob[0]->rect.left = 0;
    mvbt_battle(menu->button, 800, 0);
    sfClock_restart(menu->clock[3]);
    return (0);
}

int magic_battle(battle_t *menu, player_t *player)
{
    int dmg = (rand() % (player->magic) + 1) * 0.1 * menu->mob[1]->pvmax;
    char str[6] = { '\0' };

    if (player->mana >= 40) {
        sfText_setColor(menu->button[17]->text, sfBlue);
        sfText_setString(menu->button[17]->text, numtostr(dmg, str));
        sfText_setPosition(menu->button[17]->text, (sfVector2f){ 450, 690 });
        sfClock_restart(menu->clock[2]);
        menu->fire = 1;
        menu->mob[1]->pv -= dmg;
        player->mana -= 40;
        mvbt_battle(menu->button, 800, 0);
        sfClock_restart(menu->clock[3]);
    }
    return (0);
}

int heal_battle(battle_t *menu, player_t *player)
{
    int heal = player->pvmax * 0.3;
    char str[6] = { '\0' };

    if (player->pv == player->pvmax || player->mana < 30)
        return (0);
    if (player->pv + heal < player->pvmax)
        player->pv += heal;
    else
        player->pv = player->pvmax;
    sfText_setColor(menu->button[18]->text, sfGreen);
    sfText_setString(menu->button[18]->text, numtostr(heal, str));
    sfText_setPosition(menu->button[18]->text, (sfVector2f){ 860, 720 });
    sfClock_restart(menu->clock[2]);
    player->mana -= 30;
    mvbt_battle(menu->button, 800, 0);
    sfClock_restart(menu->clock[3]);
    return (0);
}

int defend_battle(battle_t *menu, player_t *player)
{
    player->defense += 1000;
    sfText_setColor(menu->button[18]->text, sfYellow);
    sfText_setString(menu->button[18]->text, "DEFEND !");
    sfText_setPosition(menu->button[18]->text, (sfVector2f){ 860, 720 });
    mvbt_battle(menu->button, 800, 0);
    sfClock_restart(menu->clock[3]);
    return (0);
}