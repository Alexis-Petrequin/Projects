/*
** EPITECH PROJECT, 2020
** main
** File description:
** main
*/

#include "include/mylist.h"

static void free_window(sfRenderWindow *window)
{
    sfRenderWindow_close(window);
    sfRenderWindow_destroy(window);
}

static void start(sfRenderWindow *window)
{
    sfRenderWindow_setFramerateLimit(window, 60);
    srand(time(NULL));
}

int main(int argc, char **argv)
{
    sfRenderWindow *window = fill_mywindow(1920, 1080, "my_rpg");
    player_t *player = init_player();
    menu_t *menu = init_menu();
    battle_t *battle = init_battle();
    map_t *map = init_map();
    int end = 2;

    if (!menu || !battle || !player || !map)
        return (84);
    start(window);
    while (sfRenderWindow_isOpen(window) && end != 1) {
        if (end == 3)
            end = play_battle(battle, window, player);
        if (end == 2)
            end = play_menu(menu, player, window);
        if (end == 4)
            end = play_map(window, map, player);
    }
    free_window(window);
    free_all(player, menu, battle, map);
    return (0);
}