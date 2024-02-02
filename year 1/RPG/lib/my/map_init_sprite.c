/*
** EPITECH PROJECT, 2019
** map_init_sprite
** File description:
** about init sprite
*/

#include "../../include/mylist.h"

button_t **fill_dial(button_t **dial, int y, int i, map_t *map)
{
    sfVector2f pos[3] = { { 9350, 8095 }, { 8900, 10020 }, { 8830, 7170 }};
    char text[20][200] = {
    {"Hey !\nI'm the best\nblacksmith around\n here !"},
    {"I'm also the only one\nso you don't have\na choice !\0"},
    {"Too bad I don't know\nhow to forge...\0"},
    {"Get out of my face !\nIf I built this house\nfar from the village\0"},
    {"It was to not get\nharassed by you !!!\0"}, {"I won't help you !\0"},
    {"Travis !\nwe need your help !\0"},
    {"A dozen monsters\ninvaded our isle !\0"},
    {"You're the only one\nwho can beat them !\0"} };

    sfText_setFont(dial[y][i].text, map->font);
    sfText_setCharacterSize(dial[y][i].text, 20);
    sfText_setColor(dial[y][i].text, sfBlack);
    sfRectangleShape_setTexture(dial[y][i].sprite, map->texture[7], sfFalse);
    sfRectangleShape_setSize(dial[y][i].sprite, (sfVector2f){ 180, 170 });
    sfRectangleShape_setPosition(dial[y][i].sprite, pos[y]);
    sfText_setString(dial[y][i].text, text[y * 3 + i]);
    sfText_setPosition(dial[y][i].text,
    (sfVector2f){ pos[y].x + 9, pos[y].y + 7 });
    return (dial);
}

sprit_t *init_sprite(sfTexture *t, sfVector2f pos, sfVector2f scl, int type)
{
    sprit_t *sprite = malloc(sizeof(sprit_t));

    if (!sprite)
        return (NULL);
    sprite->sprite = sfSprite_create();
    sfSprite_setTexture(sprite->sprite, t, sfTrue);
    sfSprite_setScale(sprite->sprite, scl);
    sfSprite_setPosition(sprite->sprite, pos);
    sprite->type = type;
    return (sprite);
}

int cluster_tree(sfTexture *t, sfVector2f dm, sfVector2f pos, sprit_t **sp)
{
    static int nb = 0;
    sfVector2f scale = {3, 3};
    int posx = pos.x;

    for (int y = 0; y < dm.y; y++) {
        for (int i = 0; i < dm.x; i++) {
            sp[nb] = init_sprite(t, pos, scale, 0);
            if (!sp[nb])
                return (84);
            pos.x += 100;
            nb++;
        }
        pos.x = posx;
        pos.y += 100;
    }
    return (0);
}

int clustingcluster(sfTexture **textu, sfVector2f *dim, sprit_t **scenary)
{
    sfVector2f pos[26] = { {12530, 6800}, {12430, 7500}, {14030, 7500},
    {12530, 7800}, {14030, 7800}, {12630, 8600}, {14030, 8600}, {12830, 8800},
    {14030, 8800}, {12930, 9500}, {14030, 9500}, {13030, 9800}, {14030, 9800},
    {13730, 13100}, {13030, 13600}, {13030, 14400}, {12330, 14500},
    {12430, 14400}, {12230, 14600}, {12730, 13700}, {12530, 13900} };

    for (int i = 0; i < 10; i++)
        if (cluster_tree(textu[0], dim[i], pos[i], scenary) == 84)
            return (84);
    cluster_tree(textu[0], dim[2], pos[10], scenary);
    cluster_tree(textu[0], dim[10], pos[11], scenary);
    cluster_tree(textu[0], dim[2], pos[12], scenary);
    cluster_tree(textu[0], dim[11], pos[13], scenary);
    cluster_tree(textu[0], dim[12], pos[14], scenary);
    cluster_tree(textu[0], dim[12], pos[15], scenary);
    cluster_tree(textu[0], dim[14], pos[17], scenary);
    cluster_tree(textu[0], dim[13], pos[16], scenary);
    cluster_tree(textu[0], dim[15], pos[18], scenary);
    cluster_tree(textu[0], dim[16], pos[19], scenary);
    cluster_tree(textu[0], dim[17], pos[20], scenary);
    return (0);
}

sprit_t **init_scenary(sfTexture **texture)
{
    sprit_t **scenary = malloc(sizeof(sprit_t *) * 1915);
    sfVector2f scale[4] = { {3, 3}, {0.5, 0.5}, {0.5, 0.5}, {0.6, 0.6} };
    sfVector2f dim[18] = { {43, 5}, {14, 3}, {28, 3}, {13, 8}, {28, 8},
    {12, 2}, {28, 2}, {10, 7}, {28, 7}, {9, 3}, {8, 3}, {31, 5},
    {38, 6}, {7, 5}, {6, 1}, {1, 2}, {3, 5}, {2, 3} };
    sfVector2f pos[5] = { {8800, 9900}, {9000, 7050}, {8175, 7020},
    {10245, 7520}, {9500, 7990} };

    if (!scenary)
        return (NULL);
    if (clustingcluster(texture, dim, scenary) == 84)
        return (NULL);
    scenary[1909] = init_sprite(texture[1], pos[0], scale[1], 1);
    scenary[1910] = init_sprite(texture[1], pos[1], scale[1], 1);
    scenary[1911] = init_sprite(texture[2], pos[2], scale[2], 2);
    scenary[1912] = init_sprite(texture[2], pos[3], scale[2], 2);
    scenary[1913] = init_sprite(texture[3], pos[4], scale[3], 3);
    scenary[1914] = NULL;
    return (scenary);
}