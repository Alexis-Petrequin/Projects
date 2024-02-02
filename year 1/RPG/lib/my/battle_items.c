/*
** EPITECH PROJECT, 2020
** battle_items
** File description:
** about item
*/

#include "../../include/mylist.h"

int potfire_battle(battle_t *menu, player_t *player)
{
    int dmg = 0.3 * menu->mob[1]->pvmax;
    char str[6] = { '\0' };

    if (dmg > 0 && player->nbpotfire > 0) {
        sfText_setColor(menu->button[17]->text, sfYellow);
        sfText_setString(menu->button[17]->text, numtostr(dmg, str));
        sfText_setPosition(menu->button[17]->text, (sfVector2f){ 450, 690 });
        sfClock_restart(menu->clock[2]);
        menu->mob[1]->pv -= dmg;
        menu->fire = 1;
        mvbt_battle(menu->button, 800, 0);
        player->nbpotfire -=1;
        sfClock_restart(menu->clock[3]);
    }
    return (0);
}

int potmana_battle(battle_t *menu, player_t *player)
{
    char str[6] = { '\0' };

    if (player->nbpotmana <= 0 || player->mana == player->manamax)
        return (0);
    if (player->mana + 40 >= player->manamax)
        player->mana = player->manamax;
    else
        player->mana += 40;
    sfText_setColor(menu->button[18]->text, sfBlue);
    sfText_setString(menu->button[18]->text, numtostr(40, str));
    sfText_setPosition(menu->button[18]->text, (sfVector2f){ 860, 720 });
    mvbt_battle(menu->button, 800, 0);
    player->nbpotmana -= 1;
    sfClock_restart(menu->clock[3]);
    return (0);
}

int pothp_battle(battle_t *menu, player_t *player)
{
    char str[6] = { '\0' };

    if (player->nbpothp <= 0 || player->pv == player->pvmax)
        return (0);
    if (player->pv + 50 >= player->pvmax)
        player->pv = player->pvmax;
    else
        player->pv += 50;
    sfText_setColor(menu->button[18]->text, sfGreen);
    sfText_setString(menu->button[18]->text, numtostr(50, str));
    sfText_setPosition(menu->button[18]->text, (sfVector2f){ 860, 720 });
    mvbt_battle(menu->button, 800, 0);
    player->nbpothp -= 1;
    sfClock_restart(menu->clock[3]);
    return (0);
}