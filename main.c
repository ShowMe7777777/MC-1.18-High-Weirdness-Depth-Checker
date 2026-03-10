#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <math.h>
#include "finders.h"
#include "generator.h"
#include "biomenoise.h"

pthread_mutex_t output_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    uint64_t seed;
    int x;
    int z;
    double weirdness;
} SeedData;

typedef struct {
    uint64_t seed;
    int x;
    int z;
    double coverage;
    int fill_size;
    int valid;
} BestResult;

typedef struct {
    uint64_t seed;
    int x;
    int z;
    double weirdness;
    int valid;
} BestWeirdness;

typedef struct {
    uint64_t seed;
    int x;
    int z;
    double depth;
    double depth_blocks;
    int fill_size;
    int valid;
    int large; 
} BestDepth;

typedef struct {
    SeedData *seeds;
    int start_idx;
    int end_idx;
    int thread_id;
    int mc_version;
    int seeds_checked;
    int seeds_passed;
    BestResult best_sb;
    BestResult best_lb;
    BestWeirdness best_wd;
    BestDepth best_depth;
    int w_mode;
    int d_mode;
} ThreadWork;

typedef struct { int x, z; } Point;

typedef struct {
    Point *data;
    int top;
    int capacity;
} Stack;

static void stack_push(Stack *s, int x, int z)
{
    if(s->top >= s->capacity)
    {
        s->capacity *= 2;
        s->data = realloc(s->data, sizeof(Point) * s->capacity);
    }
    s->data[s->top].x = x;
    s->data[s->top].z = z;
    s->top++;
}

static Point stack_pop(Stack *s)
{
    return s->data[--s->top];
}

#define FF_RADIUS 2000
#define DEPTH_THRESHOLD (-0.4864)
#define COVERAGE_THRESHOLD 0.07
#define COVERAGE_MIN (-2.0)
#define COVERAGE_MAX DEPTH_THRESHOLD

int flood_fill(BiomeNoise *bn, int startx, int startz)
{
    double d = sampleClimatePara(bn, NULL, startx / 4.0, startz / 4.0);
    if(d >= DEPTH_THRESHOLD)
        return 0;

    int size = 2 * FF_RADIUS + 1;
    char *visited = calloc(size * size, 1);
    if(!visited) return 0;

    Stack s;
    s.capacity = 65536;
    s.top = 0;
    s.data = malloc(sizeof(Point) * s.capacity);

    #define TO_IDX(px, pz) (((pz) - startz + FF_RADIUS) * size + ((px) - startx + FF_RADIUS))
    #define IN_BOUNDS(px, pz) ((px) >= startx - FF_RADIUS && (px) <= startx + FF_RADIUS && \
                               (pz) >= startz - FF_RADIUS && (pz) <= startz + FF_RADIUS)

    stack_push(&s, startx, startz);
    visited[TO_IDX(startx, startz)] = 1;

    int dx[] = { 1, -1,  0,  0 };
    int dz[] = { 0,  0,  1, -1 };
    int count = 0;

    while(s.top > 0)
    {
        Point p = stack_pop(&s);
        count++;

        for(int i = 0; i < 4; i++)
        {
            int nx = p.x + dx[i];
            int nz = p.z + dz[i];

            if(!IN_BOUNDS(nx, nz))
                continue;

            int idx = TO_IDX(nx, nz);
            if(visited[idx])
                continue;

            visited[idx] = 1;

            double val = sampleClimatePara(bn, NULL, nx / 4.0, nz / 4.0);
            if(val < DEPTH_THRESHOLD)
                stack_push(&s, nx, nz);
        }
    }

    free(visited);
    free(s.data);
    return count;

    #undef TO_IDX
    #undef IN_BOUNDS
}

int climateNoiseSamples(BiomeNoise *bn, int centerx, int centerz, int widthx, int widthz,
                        int stepsize, double coveragethreshold, double minvalue, double maxvalue,
                        double *coverage_out)
{
    int pointscount = 0;
    int total_points = 0;

    for(int samplesz = centerz - widthz/2; samplesz < centerz + widthz/2; samplesz += stepsize)
    {
        for(int samplesx = centerx - widthx/2; samplesx < centerx + widthx/2; samplesx += stepsize)
        {
            total_points++;
            double depthValue = sampleClimatePara(bn, NULL, samplesx/4.0, samplesz/4.0);
            if(minvalue <= depthValue && depthValue <= maxvalue)
                pointscount++;
        }
    }

    double coverage = (total_points > 0) ? (100.0 * pointscount / total_points) : 0.0;
    if(coverage_out)
        *coverage_out = coverage;

    double pointsthreshold = ((double)widthx/stepsize) * ((double)widthz/stepsize) * coveragethreshold;
    return pointscount > pointsthreshold;
}

int find_seed_point(BiomeNoise *bn, int centerx, int centerz, int widthx, int widthz,
                    int stepsize, int *out_x, int *out_z)
{
    for(int samplesz = centerz - widthz/2; samplesz < centerz + widthz/2; samplesz += stepsize)
    {
        for(int samplesx = centerx - widthx/2; samplesx < centerx + widthx/2; samplesx += stepsize)
        {
            double val = sampleClimatePara(bn, NULL, samplesx/4.0, samplesz/4.0);
            if(val < DEPTH_THRESHOLD)
            {
                *out_x = samplesx;
                *out_z = samplesz;
                return 1;
            }
        }
    }
    return 0;
}

double min_depth_in_area(BiomeNoise *bn, int cx, int cz)
{
    double min_d = 999999.0;
    for(int dz = -32; dz <= 32; dz++)
        for(int dx = -32; dx <= 32; dx++)
        {
            double v = sampleClimatePara(bn, NULL, (cx + dx) / 4.0, (cz + dz) / 4.0);
            if(v < min_d) min_d = v;
        }
    return min_d;
}

void check_seed(uint64_t seed, int x, int z, int mc_version, int *passed,
                BestResult *best_sb, BestResult *best_lb,
                double weirdness, int w_mode, int d_mode,
                BestWeirdness *best_wd, BestDepth *best_depth)
{
    if(w_mode)
    {
        if(!best_wd->valid || fabs(weirdness) > fabs(best_wd->weirdness))
        {
            best_wd->seed      = seed;
            best_wd->x         = x;
            best_wd->z         = z;
            best_wd->weirdness = weirdness;
            best_wd->valid     = 1;
        }
    }

    BiomeNoise bn;
    initBiomeNoise(&bn, mc_version);

    if(w_mode || d_mode)
    {
        setClimateParaSeed(&bn, seed, 0, NP_DEPTH, 18);
        double min_d_sb = d_mode ? min_depth_in_area(&bn, x, z) : 0.0;
        double depth_blocks_sb = min_d_sb * 76.0;

        setClimateParaSeed(&bn, seed, 1, NP_DEPTH, 8);
        double min_d_lb = d_mode ? min_depth_in_area(&bn, x, z) : 0.0;
        double depth_blocks_lb = min_d_lb * 76.0;

        int print_sb = (w_mode && fabs(weirdness) > 2.49) || (d_mode && min_d_sb < -1.4);
        int print_lb = (w_mode && fabs(weirdness) > 2.49) || (d_mode && min_d_lb < -1.4);

        char wd_str[32], depth_str[32], dblk_str[32];
        snprintf(wd_str, sizeof(wd_str), "%.6f", weirdness);

        if(print_sb || print_lb)
        {
            pthread_mutex_lock(&output_lock);
            if(print_sb)
            {
                if(w_mode && d_mode)
                {
                    snprintf(depth_str, sizeof(depth_str), "%.6f", min_d_sb);
                    snprintf(dblk_str,  sizeof(dblk_str),  "%.1f", depth_blocks_sb);
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s  %-12s  %-14s\n",
                           seed, x, z, "SB", wd_str, depth_str, dblk_str);
                }
                else if(w_mode)
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s\n",
                           seed, x, z, "SB", wd_str);
                else
                {
                    snprintf(depth_str, sizeof(depth_str), "%.6f", min_d_sb);
                    snprintf(dblk_str,  sizeof(dblk_str),  "%.1f", depth_blocks_sb);
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s  %-14s\n",
                           seed, x, z, "SB", depth_str, dblk_str);
                }
            }
            if(print_lb)
            {
                if(w_mode && d_mode)
                {
                    snprintf(depth_str, sizeof(depth_str), "%.6f", min_d_lb);
                    snprintf(dblk_str,  sizeof(dblk_str),  "%.1f", depth_blocks_lb);
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s  %-12s  %-14s\n",
                           seed, x, z, "LB", wd_str, depth_str, dblk_str);
                }
                else if(w_mode)
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s\n",
                           seed, x, z, "LB", wd_str);
                else
                {
                    snprintf(depth_str, sizeof(depth_str), "%.6f", min_d_lb);
                    snprintf(dblk_str,  sizeof(dblk_str),  "%.1f", depth_blocks_lb);
                    printf("%-22lu  %-14d  %-14d  %-10s  %-12s  %-14s\n",
                           seed, x, z, "LB", depth_str, dblk_str);
                }
            }
            fflush(stdout);
            pthread_mutex_unlock(&output_lock);
            (*passed)++;
        }

        if(d_mode && (!best_depth->valid || min_d_sb < best_depth->depth))
        {
            best_depth->seed         = seed;
            best_depth->x            = x;
            best_depth->z            = z;
            best_depth->depth        = min_d_sb;
            best_depth->depth_blocks = depth_blocks_sb;
            best_depth->large        = 0;
            best_depth->valid        = 1;
        }
        if(min_d_lb < best_depth->depth)
        {
            best_depth->seed         = seed;
            best_depth->x            = x;
            best_depth->z            = z;
            best_depth->depth        = min_d_lb;
            best_depth->depth_blocks = depth_blocks_lb;
            best_depth->large        = 1;
        }
    }
    else
    {
        double coverage = 0.0;

        setClimateParaSeed(&bn, seed, 0, NP_DEPTH, 18);
        int small = climateNoiseSamples(&bn, x, z, 800, 800, 4, COVERAGE_THRESHOLD, COVERAGE_MIN, COVERAGE_MAX, &coverage);

        if(small == 1)
        {
            int fx = x, fz = z;
            find_seed_point(&bn, x, z, 800, 800, 4, &fx, &fz);
            int fill_size = flood_fill(&bn, fx, fz);

            char cov_str[16];
            snprintf(cov_str, sizeof(cov_str), "%.2f%%", coverage);
            pthread_mutex_lock(&output_lock);
            printf("%-22lu  %-14d  %-14d  %-10s  %-10s  %-13d\n", seed, x, z, "SB", cov_str, fill_size);
            fflush(stdout);
            pthread_mutex_unlock(&output_lock);
            (*passed)++;

            if(!best_sb->valid || fill_size > best_sb->fill_size)
            {
                best_sb->seed      = seed;
                best_sb->x         = x;
                best_sb->z         = z;
                best_sb->coverage  = coverage;
                best_sb->fill_size = fill_size;
                best_sb->valid     = 1;
            }
        }

        coverage = 0.0;
        setClimateParaSeed(&bn, seed, 1, NP_DEPTH, 8);
        int large = climateNoiseSamples(&bn, x, z, 800, 800, 4, COVERAGE_THRESHOLD, COVERAGE_MIN, COVERAGE_MAX, &coverage);

        if(large == 1)
        {
            int fx = x, fz = z;
            find_seed_point(&bn, x, z, 800, 800, 4, &fx, &fz);
            int fill_size = flood_fill(&bn, fx, fz);

            char cov_str[16];
            snprintf(cov_str, sizeof(cov_str), "%.2f%%", coverage);
            pthread_mutex_lock(&output_lock);
            printf("%-22lu  %-14d  %-14d  %-10s  %-10s  %-13d\n", seed, x, z, "LB", cov_str, fill_size);
            fflush(stdout);
            pthread_mutex_unlock(&output_lock);
            (*passed)++;

            if(!best_lb->valid || fill_size > best_lb->fill_size)
            {
                best_lb->seed      = seed;
                best_lb->x         = x;
                best_lb->z         = z;
                best_lb->coverage  = coverage;
                best_lb->fill_size = fill_size;
                best_lb->valid     = 1;
            }
        }
    }
}

void* thread_worker(void* arg)
{
    ThreadWork* work = (ThreadWork*)arg;

    for(int i = work->start_idx; i < work->end_idx; i++)
    {
        check_seed(work->seeds[i].seed, work->seeds[i].x, work->seeds[i].z,
                   work->mc_version, &work->seeds_passed,
                   &work->best_sb, &work->best_lb,
                   work->seeds[i].weirdness, work->w_mode, work->d_mode,
                   &work->best_wd, &work->best_depth);
        work->seeds_checked++;
    }

    return NULL;
}

int main(int argc, char **argv)
{
    int mc_version = MC_1_21;
    char line[1024];
    FILE *input = stdin;

    int num_threads = 24;

    if(argc > 2)
    {
        num_threads = atoi(argv[2]);
        if(num_threads < 1 || num_threads > 64)
        {
            fprintf(stderr, "Invalid thread count. Using default: 24\n");
            num_threads = 24;
        }
    }

    int w_mode = 0;
    int d_mode = 0;

    int file_arg = -1;
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "--w") == 0)
            w_mode = 1;
        else if(strcmp(argv[i], "--d") == 0)
            d_mode = 1;
        else if(argv[i][0] != '-' && file_arg == -1)
            file_arg = i;
        else if(argv[i][0] != '-')
            num_threads = atoi(argv[i]);
    }

    if(file_arg == -1)
    {
        printf("Depth Checker (Multithreaded)\n");
        printf("Usage: %s <input_file> [--w] [--d] [num_threads]\n\n", argv[0]);
        printf("Default threads: 24\n\n");
        printf("  --w   Show weirdness values (filters: absv weirdness > 2.49)\n");
        printf("  --d   Show min depth values (filters: depth noise < -1.4)\n");
        printf("  Both flags can be used together\n\n");
        printf("Expected format (space-separated, one entry per line):\n");
        printf("  <seed> <x> <z> <weirdness>\n");
        printf("  8608349212741999870 0 0 2.345678\n\n");
        return 0;
    }

    input = fopen(argv[file_arg], "r");
    if(!input)
    {
        fprintf(stderr, "Error: Could not open file '%s'\n", argv[file_arg]);
        return 1;
    }

    SeedData *seeds = NULL;
    int seed_count = 0;
    int seed_capacity = 1000;
    seeds = malloc(sizeof(SeedData) * seed_capacity);

    while(fgets(line, sizeof(line), input))
    {
        if(line[0] == '\n' || line[0] == '\r' || line[0] == '#')
            continue;

        uint64_t seed;
        long x, z, ignored;

        double weirdness = 0.0;
        int parsed = sscanf(line, "%llu %ld %ld %lf",
                            (unsigned long long*)&seed, &x, &z, &weirdness);
        if(parsed < 3)
            continue;

        if(seed_count >= seed_capacity)
        {
            seed_capacity *= 2;
            seeds = realloc(seeds, sizeof(SeedData) * seed_capacity);
        }

        seeds[seed_count].seed      = seed;
        seeds[seed_count].x         = (int)x;
        seeds[seed_count].z         = (int)z;
        seeds[seed_count].weirdness = weirdness;
        seed_count++;
    }

    fclose(input);

    if(seed_count == 0)
    {
        printf("No seeds found in file!\n");
        free(seeds);
        return 1;
    }

    printf("Checking %d seeds from %s with %d threads.%s%s\n\n", seed_count, argv[file_arg], num_threads, w_mode ? " [--w]" : "", d_mode ? " [--d]" : "");
    if(w_mode && d_mode)
    {
        printf("%-22s  %-14s  %-14s  %-10s  %-12s  %-12s  %-14s\n",
               "Seed", "X", "Z", "Biome Size", "Weirdness", "Depth", "Depth (blocks)");
        printf("%-22s  %-14s  %-14s  %-10s  %-12s  %-12s  %-14s\n",
               "----------------------", "--------------", "--------------",
               "----------", "------------", "------------", "--------------");
    }
    else if(w_mode)
    {
        printf("%-22s  %-14s  %-14s  %-10s  %-12s\n",
               "Seed", "X", "Z", "Biome Size", "Weirdness");
        printf("%-22s  %-14s  %-14s  %-10s  %-12s\n",
               "----------------------", "--------------", "--------------",
               "----------", "------------");
    }
    else if(d_mode)
    {
        printf("%-22s  %-14s  %-14s  %-10s  %-12s  %-14s\n",
               "Seed", "X", "Z", "Biome Size", "Depth", "Depth (blocks)");
        printf("%-22s  %-14s  %-14s  %-10s  %-12s  %-14s\n",
               "----------------------", "--------------", "--------------",
               "----------", "------------", "--------------");
    }
    else
    {
        printf("%-22s  %-14s  %-14s  %-10s  %-10s  %-13s\n",
               "Seed", "X", "Z", "Biome Size", "Coverage", "Sinkhole Size");
        printf("%-22s  %-14s  %-14s  %-10s  %-10s  %-13s\n",
               "----------------------", "--------------", "--------------",
               "----------", "----------", "-------------");
    }

    pthread_t* threads = malloc(sizeof(pthread_t) * num_threads);
    ThreadWork* work = malloc(sizeof(ThreadWork) * num_threads);

    int seeds_per_thread = (seed_count + num_threads - 1) / num_threads;

    for(int i = 0; i < num_threads; i++)
    {
        work[i].seeds        = seeds;
        work[i].thread_id    = i;
        work[i].start_idx    = i * seeds_per_thread;
        work[i].end_idx      = (i + 1) * seeds_per_thread;
        if(work[i].end_idx > seed_count)
            work[i].end_idx = seed_count;
        work[i].mc_version   = mc_version;
        work[i].seeds_checked = 0;
        work[i].seeds_passed  = 0;
        work[i].best_sb.valid    = 0;
        work[i].best_lb.valid    = 0;
        work[i].best_wd.valid    = 0;
        work[i].best_depth.valid = 0;
        work[i].w_mode           = w_mode;
        work[i].d_mode           = d_mode;

        if(work[i].start_idx < seed_count)
        {
            if(pthread_create(&threads[i], NULL, thread_worker, &work[i]) != 0)
            {
                fprintf(stderr, "Error creating thread %d\n", i);
                return 1;
            }
        }
    }

    for(int i = 0; i < num_threads; i++)
    {
        if(work[i].start_idx < seed_count)
            pthread_join(threads[i], NULL);
    }

    int total_checked = 0, total_passed = 0;
    BestResult best_sb = {0}, best_lb = {0};

    for(int i = 0; i < num_threads; i++)
    {
        total_checked += work[i].seeds_checked;
        total_passed  += work[i].seeds_passed;

        if(work[i].best_sb.valid &&
           (!best_sb.valid || work[i].best_sb.fill_size > best_sb.fill_size))
            best_sb = work[i].best_sb;

        if(work[i].best_lb.valid &&
           (!best_lb.valid || work[i].best_lb.fill_size > best_lb.fill_size))
            best_lb = work[i].best_lb;
    }

    BestWeirdness best_wd = {0};
    BestDepth best_depth  = {0};
    for(int i = 0; i < num_threads; i++)
    {
        if(work[i].best_wd.valid &&
           (!best_wd.valid || fabs(work[i].best_wd.weirdness) > fabs(best_wd.weirdness)))
            best_wd = work[i].best_wd;

        if(work[i].best_depth.valid &&
           (!best_depth.valid || work[i].best_depth.depth < best_depth.depth))
            best_depth = work[i].best_depth;
    }

    printf("\n%d/%d seeds were valid.\n", total_passed, total_checked);

    if(w_mode)
    {
        if(best_wd.valid)
            printf("Best weirdness seed: %-22lu  %-14d  %-14d  %.6f\n",
                   best_wd.seed, best_wd.x, best_wd.z, best_wd.weirdness);
        else
            printf("Best weirdness seed: none\n");
    }
    if(d_mode)
    {
        if(best_depth.valid)
        {
            char depth_str[32], dblk_str[32];
            snprintf(depth_str, sizeof(depth_str), "%.6f", best_depth.depth);
            snprintf(dblk_str,  sizeof(dblk_str),  "%.1f", best_depth.depth_blocks);
            printf("Best depth seed:     %-22lu  %-14d  %-14d  %-10s  %-12s  %-14s\n",
                   best_depth.seed, best_depth.x, best_depth.z,
                   best_depth.large ? "LB" : "SB", depth_str, dblk_str);
        }
        else
            printf("Best depth seed: none\n");
    }
    if(!w_mode && !d_mode)
    {
        if(best_sb.valid)
        {
            char cov[16];
            snprintf(cov, sizeof(cov), "%.2f%%", best_sb.coverage);
            printf("Best small biomes seed: %-22lu  %-14d  %-14d  %-10s  %-10s  %-13d\n",
                   best_sb.seed, best_sb.x, best_sb.z, "SB", cov, best_sb.fill_size);
        }
        else
            printf("Best small biomes seed: none\n");

        if(best_lb.valid)
        {
            char cov[16];
            snprintf(cov, sizeof(cov), "%.2f%%", best_lb.coverage);
            printf("Best large biomes seed: %-22lu  %-14d  %-14d  %-10s  %-10s  %-13d\n",
                   best_lb.seed, best_lb.x, best_lb.z, "LB", cov, best_lb.fill_size);
        }
        else
            printf("Best large biomes seed: none\n");
    }

    free(threads);
    free(work);
    free(seeds);
    pthread_mutex_destroy(&output_lock);

    return 0;
}