#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

//for atgument reading
static int centroid_num = 0;            //number of clusters
static int star_num = 0;                //number of stars

//for initialization
static double max_x = -DBL_MIN;
static double min_x = DBL_MAX;
static double max_y = -DBL_MIN;
static double min_y = DBL_MAX;
static double max_z = -DBL_MIN;
static double min_z = DBL_MAX;

//for cluster data record
static double * stars_xyz = NULL;    //the xyz star coordination data
static double * stars_lbr = NULL;    //the lbr coordination orignal data
static int * assignment = NULL;      //identify every star with a cluster number
static int * cluster_sz = NULL;      //size of each cluster

//for kmeans iteration
static double * centroids = NULL;

double dist_2(double x, double y, double z, double a, double b, double c)
{
    return (x-a)*(x-a)+(y-b)*(y-b)+(z-c)*(z-c);
}

//Lloyd's algorithm
int lloyd_kmeans(double s_xyz[], int s_num, double c_xyz[], int c_num, int assi[], int c_szs[])
{
    int loop_counter = 1;
    double sse = 0.0;
    struct timeval tik, toc;
    double * c_avg = (double *)malloc(sizeof(double)*c_num*4);
    memset(c_avg, 0, sizeof(double)*c_num*4);
    //record start time point, tik
    gettimeofday(&tik, NULL);

    //loop and update centroid and sse until convergence
    while(1)
    {
        //reset statistic
        memset(c_avg, 0, sizeof(double)*c_num*4);
        //update cluster number of each star.
        //select the nearest centroid as star's new cluster center.
        for(int s_id = 0; s_id < s_num; ++s_id)
        {
            double min_dist_2 = DBL_MAX;
            for(int c_id = 0; c_id < c_num; ++c_id)
            {
                //calulate dist
                double my_dist_2 = dist_2(c_xyz[c_id*3+0], c_xyz[c_id*3+1], c_xyz[c_id*3+2], s_xyz[s_id*3+0], s_xyz[s_id*3+1], s_xyz[s_id*3+2]);
                //if star tested is closer to this centroid, update its cluster number stored in assi
                if(my_dist_2 < min_dist_2)
                {
                    //update minima dist
                    min_dist_2 = my_dist_2;
                    //update the assignment array
                    assi[s_id] = c_id;
                }
            }
            c_avg[assi[s_id]*4+0] += s_xyz[s_id*3+0];
            c_avg[assi[s_id]*4+1] += s_xyz[s_id*3+1];
            c_avg[assi[s_id]*4+2] += s_xyz[s_id*3+2];
            c_avg[assi[s_id]*4+3] += 1.0;
        }

        //update centroid based on initial cluster result;
        for(int cnt = 0; cnt < c_num; ++cnt)
        {
            if(0 != c_avg[cnt*4+3])
            {
                c_xyz[cnt*3+0] = c_avg[cnt*4+0] / c_avg[cnt*4+3];
                c_xyz[cnt*3+1] = c_avg[cnt*4+1] / c_avg[cnt*4+3];
                c_xyz[cnt*3+2] = c_avg[cnt*4+2] / c_avg[cnt*4+3];
            }
        }

        //calculate new sse and update count of each cluster
        double new_sse = 0.0;
        memset(c_szs, 0, sizeof(int)*c_num);
        for(int cnt = 0; cnt < s_num; ++cnt)
        {
            new_sse += dist_2(s_xyz[cnt*3+0], s_xyz[cnt*3+1], s_xyz[cnt*3+2], c_xyz[assi[cnt]*3+0], c_xyz[assi[cnt]*3+1], c_xyz[assi[cnt]*3+2]);
            ++c_szs[assi[cnt]];
        }
        double conv = fabs(new_sse - sse);
        sse = new_sse;

        printf("\nK-Means Centroid Now(%dth iter) delta_sse = %lf:\n", loop_counter, conv);
        for(int cnt = 0; cnt < c_num; ++cnt)
        {
            printf("  c%d(%.5lf,%.5lf,%.5lf) ", cnt, c_xyz[cnt*3+0], c_xyz[cnt*3+1], c_xyz[cnt*3+2]);
        }

        ++loop_counter;

        //break if convergence or update sse
        if(conv < 0.001) break;
    }

    //toc when convergence
    gettimeofday(&toc, NULL);

    //printing message and logging
    printf("\nCluster assignmet:\n");
    for(int cnt = 0; cnt < c_num; ++cnt)
    {
        printf("\tc%d[%.5lf,%.5lf,%.5lf] with %d in %d stars %.2lf%%\n", cnt, c_xyz[cnt*3+0], c_xyz[cnt*3+1], c_xyz[cnt*3+2], c_szs[cnt], s_num, c_szs[cnt]*100.0/s_num);
    }
    printf("\nTime elapsed: %10.6lfs\n", (toc.tv_sec - tik.tv_sec) + (toc.tv_usec - tik.tv_usec)/1000000.0);

    free(c_avg);

    return 0;
}

int main(int argc, char* argv[])
{
    //step-0 parse argument and do some inite
    int file_num = 0;    //number of star file
    if(argc > 2)
    {
        //handle cluster_num
        centroid_num = atoi(argv[1]);
        printf("%d\n", centroid_num);
        //handle star files
        for(int idx = 2; idx < argc; ++idx)
        {
            if(strlen(argv[idx]) > 0)
            {
                FILE * fp = fopen(argv[idx], "r");
                int digits = 0;
                if(NULL != fp)
                {
                    fscanf(fp, "%d", &digits);
                    star_num += digits;
                    file_num++;
                }
                fclose(fp);
            }
        }

        //print errors or messages
        printf("Arguments succesfully parsed.\n");
        printf("\tstar file number: %d\n", file_num);
        printf("\tstar number: %d\n", star_num);
        printf("\tcluster number: %d\n", centroid_num);
        if(centroid_num < 1)
        {
            printf("\tERROR: Strange cluster num [%d]... ABORT\n", centroid_num);
            return 1;
        }
        if(0 == file_num)
        {
            printf("\tERROR: No valid star files... ABORT\n");
            return 1;
        }
        if(0 == star_num)
        {
            printf("\tERROR: No stars to be clustered... ABORT\n");
            return 1;
        }
    }
    else
    {
        printf("Wrong arguments... ABORT!\n");
        return 1;
    }

    //step-1 Allocate memory space for star data
    stars_xyz = (double *)malloc(sizeof(double)*star_num*3);
    memset(stars_xyz, 0, sizeof(double)*star_num*3);
    stars_lbr = (double *)malloc(sizeof(double)*star_num*3);
    memset(stars_lbr, 0, sizeof(double)*star_num*3);
    centroids = (double *)malloc(sizeof(double)*centroid_num*3);
    memset(centroids, 0, sizeof(double)*centroid_num*3);
    cluster_sz = (int *)malloc(sizeof(int)*centroid_num);
    memset(cluster_sz, 0, sizeof(int)*centroid_num);
    assignment = (int *)malloc(sizeof(int)*star_num);
    memset(assignment, 0, sizeof(int)*star_num);

    //step-2 read star files
    int occu = 0;
    double l = 0.0, b = 0.0, r = 0.0;
    int read_num = 0;           //number read from star file
    //open every star file to read
    for(int idx = 2; idx < argc; ++idx)
    {
        if(0 < strlen(argv[idx]))
        {
            //open star file
            FILE * fp = fopen(argv[idx], "r");
            //ommit bad files
            if(NULL == fp) continue;
            //fetch and transform star datas
            fscanf(fp, "%d", &occu);
            while(EOF != fscanf(fp, "%lf %lf %lf", &l, &b, &r))
            {
                //record input stars
                stars_lbr[read_num * 3 + 0] = l;
                stars_lbr[read_num * 3 + 1] = b;
                stars_lbr[read_num * 3 + 2] = r;
                //convert lbr (galactic) to x y z (cartesian) [formula from star_visualizier]
                l = l * (3.14159265358979323846 / 180);
                b = b * (3.14159265358979323846 / 180);
                stars_xyz[read_num * 3 + 0] = r * cos(b) * sin(l);;
                stars_xyz[read_num * 3 + 1] = r * cos(l) * cos(b);
                stars_xyz[read_num * 3 + 2] = r * sin(b);
                //do statistics
                min_x = stars_xyz[read_num * 3 + 0] < min_x ? stars_xyz[read_num * 3 + 0] : min_x;
                max_x = stars_xyz[read_num * 3 + 0] > max_x ? stars_xyz[read_num * 3 + 0] : max_x;
                min_y = stars_xyz[read_num * 3 + 1] < min_y ? stars_xyz[read_num * 3 + 1] : min_y;
                max_y = stars_xyz[read_num * 3 + 1] > max_y ? stars_xyz[read_num * 3 + 1] : max_y;
                min_z = stars_xyz[read_num * 3 + 2] < min_z ? stars_xyz[read_num * 3 + 2] : min_z;
                max_z = stars_xyz[read_num * 3 + 2] > max_z ? stars_xyz[read_num * 3 + 2] : max_z;
                read_num++;
            }
            fclose(fp);
        }
    }
    //print logs
    printf("Stars succesfully loaded.\n");
    printf("\tstars exist: %d\n", star_num);
    printf("\tstars read: %d\n", read_num);
    printf("\tmin / max x val: %lf / %lf\n", min_x, max_x);
    printf("\tmin / max y val: %lf / %lf\n", min_y, max_y);
    printf("\tmin / max z val: %lf / %lf\n", min_z, max_z);

    //generate random centroids and random cluster for kmeans algorithm
    for(int cnt = 0; cnt < centroid_num; ++cnt)
    {
        srand(cnt*100+13);
        centroids[cnt*3+0] = (max_x - min_x)*(rand()%10/10.0) + min_x;
        srand(cnt*200+31);
        centroids[cnt*3+1] = (max_y - min_y)*(rand()%10/10.0) + min_y;
        srand(cnt*250+47);
        centroids[cnt*3+2] = (max_z - min_z)*(rand()%10/10.0) + min_z;
    }
    printf("K-Means Centroid Init:\n");
    for(int cnt = 0; cnt < centroid_num; ++cnt)
    {
        printf("\tc%d(%.5lf,%.5lf,%.5lf) ", cnt, centroids[cnt*3+0], centroids[cnt*3+1], centroids[cnt*3+2]);
        if(cnt % 3 == 1) printf("\n");
    }

    //step-4 call non-parallel k-means clustering
    lloyd_kmeans(stars_xyz, star_num, centroids, centroid_num, assignment, cluster_sz);

    //write non-mpi kmeans result to files
    system("rm -rf ./clusters_non_mpi");
    mkdir("./clusters_non_mpi", 0777);
    FILE ** fa = (FILE **)malloc(sizeof(FILE*)*centroid_num);
    for(int cnt = 0; cnt < centroid_num; ++cnt)
    {
        fa[cnt] = NULL;
    }
    char fn[128] = {'\0'};
    //create and open files
    for(int cnt = 0; cnt < centroid_num; ++cnt)
    {
        if(0 != cluster_sz[cnt])
        {
            sprintf(fn, "./clusters_non_mpi/stars-%d.txt", cnt);
            fa[cnt] = fopen(fn, "w");
            fprintf(fa[cnt], "%d\n", cluster_sz[cnt]);
        }
    }
    //write data of each cluster, Ommit the empty file
    for(int cnt = 0; cnt < star_num; ++cnt)
    {
        if(NULL != fa[assignment[cnt]])
        {
            fprintf(fa[assignment[cnt]], "%f %f %f\n", stars_lbr[cnt*3+0], stars_lbr[cnt*3+1], stars_lbr[cnt*3+2]);
        }
    }
    //close files
    for(int cnt = 0; cnt < centroid_num; cnt++)
    {
        if(NULL != fa[cnt])
        {
            fclose(fa[cnt]);
        }
    }
    free(fa);

    //dellocate memory
    free(stars_xyz); stars_xyz = NULL;
    free(stars_lbr); stars_lbr = NULL;
    free(centroids); centroids = NULL;
    free(assignment); assignment = NULL;
    free(cluster_sz); cluster_sz = NULL;

    return 0;
}
