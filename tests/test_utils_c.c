/*
 * test_utils_c.c - C 层回归测试（P0 安全修复 + 别名安全）
 *
 * 编译：
 *   gcc tests/test_utils_c.c src/utils.c -Isrc -Iinclude -o test_utils_c.exe -lm
 * 运行：
 *   ./test_utils_c.exe
 *
 * 覆盖：
 *   1. replace_image_to_label 在 "output 与 input 同一缓冲"（别名）下的正确性
 *      —— 这正是实测中因 output[0]='\0' 提前清空输入而回归的场景
 *   2. find_replace / find_replace_extension 的别名调用
 *   3. 超长输入（> 原 8192 上限）不再溢出且结果正确
 *   4. safe_strcat 严格按 size 截断，不越界
 *   5. trim 对全空白串不再越界读
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"

static int failures = 0;
static int total = 0;

static void check(const char *name, int cond, const char *detail)
{
    ++total;
    if (cond) {
        printf("  [OK]   %s\n", name);
    } else {
        printf("  [FAIL] %s  -> %s\n", name, detail ? detail : "");
        ++failures;
    }
}

int main(void)
{
    /* ---------- 1. replace_image_to_label 别名调用 ---------- */
    printf("[1] replace_image_to_label（output 与 input 别名）\n");
    {
        char p[4096] = "C:\\Users\\me\\dataset\\images\\Control_Snap-16950__t016.png";
        replace_image_to_label(p, p);            /* in-place，别名 */
        check("同目录 png -> txt",
              strcmp(p, "C:\\Users\\me\\dataset\\images\\Control_Snap-16950__t016.txt") == 0, p);
    }
    {
        char p[4096] = "/data/images/train2017/img_001.jpg";
        replace_image_to_label(p, p);
        check("COCO images/train2017 -> labels/train2017 + .txt",
              strcmp(p, "/data/labels/train2017/img_001.txt") == 0, p);
    }
    {
        char p[4096] = "/voc/VOC2007/JPEGImages/img_002.bmp";
        replace_image_to_label(p, p);
        check("PascalVOC JPEGImages -> labels + .txt",
              strcmp(p, "/voc/VOC2007/labels/img_002.txt") == 0, p);
    }

    /* ---------- 2. find_replace 别名调用 ---------- */
    printf("\n[2] find_replace / find_replace_extension（别名）\n");
    {
        char s[64] = "a/b/c.png";
        find_replace(s, "b", "B", s);
        check("find_replace 同缓冲替换", strcmp(s, "a/B/c.png") == 0, s);
    }
    {
        char s[64] = "pic.jpeg";
        find_replace_extension(s, ".jpeg", ".txt", s);
        check("find_replace_extension 同缓冲替换", strcmp(s, "pic.txt") == 0, s);
    }
    {
        char s[64] = "pic.bmp";
        find_replace_extension(s, ".png", ".txt", s);
        check("扩展名不匹配时原样返回", strcmp(s, "pic.bmp") == 0, s);
    }

    /* ---------- 3. 超长输入不溢出 ---------- */
    printf("\n[3] 超长输入（远超原 8192 上限）\n");
    {
        /* 12000 字符的路径；原实现 sprintf 到 8192 缓冲会溢出 */
        char *long_in = (char*)calloc(12001, 1);
        memset(long_in, 'a', 12000);
        long_in[11996] = '.';
        long_in[11997] = 'p'; long_in[11998] = 'n'; long_in[11999] = 'g';

        char *out = (char*)calloc(12001, 1);
        find_replace_n(long_in, "zzz_no_such", "R", out, 12001);
        check("12000 字符路径 find_replace_n 不溢出且内容一致",
              strcmp(out, long_in) == 0, "长度或内容不符");

        /* replace_image_to_label_n 使用显式 output_size */
        char *lbl = (char*)calloc(12001, 1);
        replace_image_to_label_n(long_in, lbl, 12001);
        check("12000 字符路径标签名以 .txt 结尾",
              strlen(lbl) == 12000 && strcmp(lbl + strlen(lbl) - 4, ".txt") == 0,
              "结尾不正确");
        free(long_in); free(out); free(lbl);
    }

    /* ---------- 4. safe_strcat 边界 ---------- */
    printf("\n[4] safe_strcat 严格按 size 截断\n");
    {
        char buf[16] = "12345";
        safe_strcat(buf, sizeof(buf), "6789");
        check("未超容量时正常追加", strcmp(buf, "123456789") == 0, buf);

        char b2[8] = "12345";
        safe_strcat(b2, sizeof(b2), "6789");
        check("超容量时被截断到 size-1", strcmp(b2, "1234567") == 0, b2);
        check("缓冲区 NUL 结尾", b2[7] == '\0', "未正确结尾");

        char b3[8] = "1234567";   /* 已占满 7 字符 */
        safe_strcat(b3, sizeof(b3), "X");
        check("已满时不再写入", strcmp(b3, "1234567") == 0, b3);
    }

    /* ---------- 5. trim 边界 ---------- */
    printf("\n[5] trim 边界情况\n");
    {
        char a[32] = "   hello world   ";
        trim(a);
        check("两端空白被去除", strcmp(a, "hello world") == 0, a);
    }
    {
        char a[32] = "     ";      /* 全空白：原实现会越界读 */
        trim(a);
        check("全空白串变成空串", strcmp(a, "") == 0, a);
    }
    {
        char a[32] = "";
        trim(a);
        check("空串不崩溃", strcmp(a, "") == 0, a);
    }

    printf("\n========================================\n");
    printf("通过 %d 项，失败 %d 项\n", total - failures, failures);
    if (failures) {
        printf("失败项：\n");
    }
    return failures ? 1 : 0;
}
