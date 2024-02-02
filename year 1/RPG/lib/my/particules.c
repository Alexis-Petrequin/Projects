/*
** EPITECH PROJECT, 2020
** particules
** File description:
** about particules
*/

#include "../../include/mylist.h"

void set_particles(sfUint8 *pixel, sfUint8 red, sfUint8 green, sfUint8 blue)
{
    pixel[0] = red;
    pixel[1] = green;
    pixel[2] = blue;
    pixel[3] = 255;
}

void reset_particles(sfUint8 *pixel)
{
    pixel[0] = 0;
    pixel[1] = 0;
    pixel[2] = 0;
    pixel[3] = 0;
}

void copy_particles(sfUint8 *from, sfUint8 *to)
{
    to[0] = from[0];
    to[1] = from[1];
    to[2] = from[2];
    to[3] = from[3];
}

int particles_comp(sfUint8 *pixel, sfUint8 red, sfUint8 green, sfUint8 blue)
{
    if (pixel[0] == red && pixel[1] == green && pixel[2] == blue)
        return (1);
    return (0);
}