/*
** EPITECH PROJECT, 2020
** battle_fire
** File description:
** about fire particules
*/

#include "../../include/mylist.h"

void up_fire_part(sfUint8 *buffer, size_t i)
{
    if (particles_comp(&buffer[i], 255, 130, 25)) {
        buffer[i + 3] -= 10;
        if (i > 400 * 4 * (rand() % 40 + 1))
            copy_particles(&buffer[i], &buffer[i - 400 * 4]);
    reset_particles(&buffer[i]);
    }
}

void mid_fire_part(sfUint8 *buffer, size_t i)
{
    int range = 400 * 4 * (rand() % 4 + 1);
    int range2 = (400 * 4) - (rand() % 3 - 1) * 4;

    if (particles_comp(&buffer[i], 255, 71, 23)) {
        buffer[i + 3] -= 1;
        if (buffer[i + 3] <= 100) {
            reset_particles(&buffer[i]);
            if ((rand() % 100) == 0)
                set_particles(&buffer[i - range], 255, 130, 25);
        }
        else if (i > 220 * 4)
            copy_particles(&buffer[i], &buffer[i - range2]);
    }
}

void botom_fire_part(sfUint8 *buffer, size_t i)
{
    int tmp = 0;

    buffer[i + 3] -= 1;
    if (buffer[i + 3] == 253) {
        if (rand() % 4) {
            tmp = (rand() % 5 - 2) * 4;
            copy_particles(&buffer[i], &buffer[i - 400 * 4 + tmp]);
            buffer[i - 400 * 4 + tmp + 3] = 255;
        } else
            set_particles(&buffer[i - 400 * 4], 255, 71, 23);
    }
    else if (buffer[i + 3] <= 100)
        reset_particles(&buffer[i]);
}

int update_particles(sfUint8 *buffer, battle_t *menu)
{
    sfTime time = sfClock_getElapsedTime(menu->clock[2]);
    int start = (400 * 300 * 4) - (rand() % 210 * 4 + 400);

    if (time.microseconds / 1000000.0 < 3.00) {
        set_particles(&buffer[start], 255, 205, 23);
        set_particles(&buffer[start], 255, 205, 23);
        for (size_t i = 0; i < 400 * 300 * 4; i += 4) {
            up_fire_part(buffer, i);
            if (particles_comp(&buffer[i], 255, 205, 23))
                botom_fire_part(buffer, i);
            mid_fire_part(buffer, i);
        }
        time = sfClock_getElapsedTime(menu->clock[2]);
        return (0);
    }
    menu->fire = 0;
}

void draw_fire(sfRenderWindow *window, sfUint8 *buffer)
{
    sfTexture *texture = sfTexture_create(400, 300);
    sfSprite *sprite = sfSprite_create();

    sfTexture_updateFromPixels(texture, buffer, 400, 300, 0, 0);
    sfSprite_setTexture(sprite, texture, sfFalse);
    sfSprite_setPosition(sprite, (sfVector2f){ 130, 523 } );
    sfSprite_setScale(sprite, (sfVector2f){ 1 , 1.3 });
    sfRenderWindow_drawSprite(window, sprite, NULL);
    sfSprite_destroy(sprite);
    sfTexture_destroy(texture);
}