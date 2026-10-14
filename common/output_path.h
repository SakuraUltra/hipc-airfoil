#ifndef HIPC_OUTPUT_PATH_H
#define HIPC_OUTPUT_PATH_H

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* Create only parent directories; the caller still checks fopen/fclose. */
static int hipc_make_parent_directories(const char *filename) {
    char path[1024];
    if (strlen(filename) >= sizeof(path)) {
        fprintf(stderr, "Error: output path is too long.\n");
        return -1;
    }
    strcpy(path, filename);
    for (char *part = path + 1; *part; ++part) {
        if (*part != '/') continue;
        *part = '\0';
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            perror(path);
            return -1;
        }
        *part = '/';
    }
    return 0;
}

#endif
