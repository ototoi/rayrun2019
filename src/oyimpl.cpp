//
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
//
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <array>
//
#include "rayrun.hpp"
#include "aligned_vector.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#if defined(_OPENMP) && !defined(NO_OMP)
#include <omp.h>
#endif

//
BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}



#define DIV_BIT 10
#define DIV_NUM 1024

#define MIN_FACE 4

static const int STACK_SIZE = ((int)(sizeof(size_t) * 16));

static float Ci_ = 0.8f;
static float Ct_ = 1.0f;
static int Iter_ = 3;

static inline float Ci() { return Ci_; }
static inline float Ct() { return Ct_; }
static inline int Iter() { return Iter_; }

typedef glm::vec3 vec3;

typedef struct triangle_t
{
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 e1;//p0 - p1
    glm::vec3 e2;//p0 - p1
    glm::vec3 center;
    int32_t face_id;
} triangle_t;

typedef const triangle_t* PCFACE;

typedef struct mesh_t
{
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<triangle_t> triangles;
} mesh_t;

typedef struct isect_t
{
    float t;//
    float u;
    float v;
    //float isect[3];
    //float ns[3];
    int32_t faceid;
} isect_t;

typedef struct ray_t
{
    vec3 org;
    vec3 dir;
    vec3 idir;
    int sign[3];
} ray_t;

static inline vec3 tovec(const float f[3])
{
    return vec3(f[0], f[1], f[2]);
}

static inline vec3 get_center(PCFACE face)
{
    return face->center;
    //return (face->p0 + face->p1 + face->p2) * (float(1) / 3);
}


//For float
static inline int test_AABB(
    const __m128 bboxes[2][3], //4boxes : min-max[2] of xyz[3] of boxes[4]
    const __m128 org[3],       //ray origin
    const __m128 idir[3],      //ray inveresed direction
    const int sign[3],         //ray xyz direction -> +:0,-:1
    __m128 tmin, __m128 tmax   //ray range tmin-tmax
)
{
    // x coordinate
    tmin = _mm_max_ps(
        tmin,
        _mm_mul_ps(_mm_sub_ps(bboxes[sign[0]][0], org[0]), idir[0]));
    tmax = _mm_min_ps(
        tmax,
        _mm_mul_ps(_mm_sub_ps(bboxes[1 - sign[0]][0], org[0]), idir[0]));

    // y coordinate
    tmin = _mm_max_ps(
        tmin,
        _mm_mul_ps(_mm_sub_ps(bboxes[sign[1]][1], org[1]), idir[1]));
    tmax = _mm_min_ps(
        tmax,
        _mm_mul_ps(_mm_sub_ps(bboxes[1 - sign[1]][1], org[1]), idir[1]));

    // z coordinate
    tmin = _mm_max_ps(
        tmin,
        _mm_mul_ps(_mm_sub_ps(bboxes[sign[2]][2], org[2]), idir[2]));
    tmax = _mm_min_ps(
        tmax,
        _mm_mul_ps(_mm_sub_ps(bboxes[1 - sign[2]][2], org[2]), idir[2]));

    return _mm_movemask_ps(_mm_cmpge_ps(tmax, tmin));
}

static const size_t EMPTY_MASK = size_t(-1);        //0xFFFF...
static const size_t SIGN_MASK = ~(EMPTY_MASK >> 1); //0x8000...

inline bool IsLeaf(size_t i)
{
    return (SIGN_MASK & i) ? true : false;
}
inline bool IsBranch(size_t i)
{
    return !IsLeaf(i);
}
inline bool IsEmpty(size_t i)
{
    return i == EMPTY_MASK;
}

inline size_t MakeLeafIndex(size_t i)
{
    return SIGN_MASK | i;
}

inline size_t GetFaceFirst(size_t i)
{
    return (~SIGN_MASK) & i;
}

#pragma pack(push, 4)
struct SIMDBVHNode
{
    __m128 bboxes[2][3]; //768
    size_t children[4];  //128
    int axis_top;        //128
    int axis_right;
    int axis_left;
    int reserved;
};
#pragma pack(pop)

//visit order table
//8*16 = 128
static const int OrderTable[] = {
    0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444,
    0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440,
    0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441,
    0x44401, 0x44401, 0x44410, 0x44410, 0x44401, 0x44401, 0x44410, 0x44410,
    0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442,
    0x44402, 0x44402, 0x44402, 0x44402, 0x44420, 0x44420, 0x44420, 0x44420,
    0x44412, 0x44412, 0x44412, 0x44412, 0x44421, 0x44421, 0x44421, 0x44421,
    0x44012, 0x44012, 0x44102, 0x44102, 0x44201, 0x44201, 0x44210, 0x44210,
    0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443,
    0x44403, 0x44403, 0x44403, 0x44403, 0x44430, 0x44430, 0x44430, 0x44430,
    0x44413, 0x44413, 0x44413, 0x44413, 0x44431, 0x44431, 0x44431, 0x44431,
    0x44013, 0x44013, 0x44103, 0x44103, 0x44301, 0x44301, 0x44310, 0x44310,
    0x44423, 0x44432, 0x44423, 0x44432, 0x44423, 0x44432, 0x44423, 0x44432,
    0x44023, 0x44032, 0x44023, 0x44032, 0x44230, 0x44320, 0x44230, 0x44320,
    0x44123, 0x44132, 0x44123, 0x44132, 0x44231, 0x44321, 0x44231, 0x44321,
    0x40123, 0x40132, 0x41023, 0x41032, 0x42301, 0x43201, 0x42310, 0x43210,
};


static void get_minmax(vec3& pmin, vec3& pmax, PCFACE face)
{
    static const float far_ = std::numeric_limits<float>::max();

    vec3 min = vec3(+far_, +far_, +far_);
    vec3 max = vec3(-far_, -far_, -far_);

    vec3 points[3];
    points[0] = face->p0;
    points[1] = face->p1;
    points[2] = face->p2;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            min[j] = std::min(min[j], points[i][j]);
            max[j] = std::max(max[j], points[i][j]);
        }
    }

    pmin = min;
    pmax = max;
}

struct code_face
{
    uint32_t code;
    PCFACE pFace;
};

struct InterNode
{
    size_t nodes[2];
    PCFACE pFace;
    size_t fsz;
    int dim;
    int dummy__;
};

typedef std::vector<code_face>::iterator code_face_iter;

struct separator
{
    separator(int level)
    {
        int p = 3 * DIV_BIT - 1 - level;
        nMask_ = 1 << p;
    }
    inline bool operator()(const code_face& cf) const
    {
        return (nMask_ & cf.code) == 0;
    }

    uint32_t nMask_;
};

static inline int GetDim(int level)
{
    return level % 3;
}

static inline uint32_t partby2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    return n;
}

static inline __m128i partby2(__m128i n)
{
    static const __m128i C0 = _mm_set1_epi32(0xff0000ff);
    static const __m128i C1 = _mm_set1_epi32(0x0300f00f);
    static const __m128i C2 = _mm_set1_epi32(0x030c30c3);
    static const __m128i C3 = _mm_set1_epi32(0x09249249);

    n = _mm_and_si128(_mm_xor_si128(n, _mm_slli_epi32(n, 16)), C0);
    n = _mm_and_si128(_mm_xor_si128(n, _mm_slli_epi32(n, 8)), C1);
    n = _mm_and_si128(_mm_xor_si128(n, _mm_slli_epi32(n, 4)), C2);
    n = _mm_and_si128(_mm_xor_si128(n, _mm_slli_epi32(n, 2)), C3);

    return n;
}

static inline uint32_t get_morton_code_SIMD(__m128i x)
{
#ifdef _MSC_VER
    _declspec(align(16)) uint32_t data[4];
#else
    __attribute__((aligned(16))) uint32_t data[4];
#endif
    _mm_store_si128((__m128i*)data, partby2(x));
    return (data[0] << 2) | (data[1] << 1) | (data[2]);
}


static inline uint32_t get_morton_code_SIMD(__m128 fp, __m128 fmin, __m128 frev)
{
    //__m128i nRet = _mm_cvtps_epi32(_mm_mul_ps(_mm_set1_ps((float)DIV_NUM),_mm_div_ps(_mm_sub_ps(fp,fmin),_mm_sub_ps(fmax,fmin))));
    __m128i nRet = _mm_cvtps_epi32(_mm_mul_ps(_mm_set1_ps((float)DIV_NUM), _mm_mul_ps(_mm_sub_ps(fp, fmin), frev)));
    return get_morton_code_SIMD(nRet);
}

static inline void create_morton_code_(unsigned int* codes, const PCFACE* faces, size_t fsz, const float min[3], const float max[3])
{
    __m128 fmin = _mm_setr_ps(min[0], min[1], min[2], 0);
    __m128 fmax = _mm_setr_ps(max[0], max[1], max[2], 0);
    __m128 frev = _mm_rcp_ps(_mm_sub_ps(fmax, fmin));

    for (size_t i = 0; i < fsz; i++)
    {
        vec3 c = get_center(faces[i]);
        __m128 fp = _mm_setr_ps((float)c[0], (float)c[1], (float)c[2], 0);
        codes[i] = get_morton_code_SIMD(fp, fmin, frev);
    }
}

void create_morton_code(std::vector<uint32_t>& codes, const PCFACE* faces, size_t sz, const vec3& min, const vec3& max)
{
    create_morton_code_(&codes[0], faces, sz, glm::value_ptr(min), glm::value_ptr(max));
}

void get_morton_bound(vec3& tmin, vec3& tmax, const vec3& min, const vec3& max)
{
    vec3 center = (min + max) * 0.5f;
    vec3 extend = (max - min) * 0.5f;

    float wid = extend[0];
    if (wid < extend[1]) wid = extend[1];
    if (wid < extend[2]) wid = extend[2];
    wid += +std::numeric_limits<float>::epsilon() * 1024;
    tmin = center - vec3(wid, wid, wid);
    tmax = center + vec3(wid, wid, wid);
}

static void CreateCodeFace(std::vector<code_face>& codes, const vec3& min, const vec3& max, PCFACE* begin, PCFACE* end)
{
    //std::vector<PCFACE> faces(begin, end);
    //std::random_shuffle(faces.begin(), faces.end());
    size_t sz = end - begin; //faces.size();
    std::vector<uint32_t> cx(sz);
    create_morton_code(cx, begin, sz, min, max);
    codes.resize(sz);
    for (size_t i = 0; i < sz; i++)
    {
        codes[i].pFace = begin[i];
        codes[i].code = cx[i];
    }
}

static inline size_t GetFaceSize(const std::vector<InterNode>& nodes, size_t index)
{
    //if(index == EMPTY_MASK)return 0;
    //if(nodes.size()<=index)return 0;
    const InterNode& node = nodes[index];
    return node.fsz;
}

static inline void GetFaces(std::vector<PCFACE>& faces, const std::vector<InterNode>& nodes, size_t index)
{
    if (index == EMPTY_MASK) return;
    //if(nodes.size()<=index)return;
    const InterNode& node = nodes[index];
    if (node.pFace)
    {
        faces.push_back(node.pFace);
    }
    else
    {
        GetFaces(faces, nodes, node.nodes[0]);
        GetFaces(faces, nodes, node.nodes[1]);
    }
}

static inline void GetMinMax(vec3& min, vec3& max, PCFACE pFace)
{
    vec3 points[] = { pFace->p0, pFace->p1, pFace->p2 };
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            min[j] = std::min(min[j], points[i][j]);
            max[j] = std::max(max[j], points[i][j]);
        }
    }
}

static inline void GetMinMax(vec3& min, vec3& max, const std::vector<InterNode>& nodes, size_t index)
{
    if (index == EMPTY_MASK) return;
    if (nodes.size() <= index) return;
    const InterNode& node = nodes[index];
    if (node.pFace)
    {
        GetMinMax(min, max, node.pFace);
    }
    else
    {
        GetMinMax(min, max, nodes, node.nodes[0]);
        GetMinMax(min, max, nodes, node.nodes[1]);
    }
}

static inline void GetMinMax(vec3& min, vec3& max, const std::vector<PCFACE>& faces)
{
    const PCFACE* f = &faces[0];
    size_t sz = faces.size();
    for (size_t i = 0; i < sz; i++)
    {
        GetMinMax(min, max, f[i]);
    }
}

static inline float GetArea(const vec3& min, const vec3& max)
{
    vec3 wid = max - min;
    return wid[0] * wid[1] + wid[1] * wid[2] + wid[2] * wid[0];
}

static inline size_t CreateInterNodeDeep(std::vector<InterNode>& nodes, code_face_iter a, code_face_iter b, int level)
{
    size_t sz = b - a;
    if (sz == 0) return EMPTY_MASK;
    if (sz == 1)
    {
        size_t offset = nodes.size();
        InterNode node;
        node.dim = GetDim(level);
        node.pFace = a->pFace;
        node.fsz = 1;
        node.nodes[0] = EMPTY_MASK;
        node.nodes[1] = EMPTY_MASK;
        nodes.push_back(node);
        return offset;
    }
    else
    {
        size_t offset = nodes.size();
        InterNode tmp;
        nodes.push_back(tmp);
        InterNode& node = nodes.back();

        node.dim = GetDim(level);
        node.pFace = NULL;
        node.fsz = sz;

        code_face_iter c = b; //
        if (level < 3 * DIV_BIT)
        {
            c = std::partition(a, b, separator(level));
        }
        else
        {
            c = a + (sz >> 1);
        }
        code_face_iter iters[] = { a, c, c, b };

        size_t sub_indices[2] = {};
        sub_indices[0] = CreateInterNodeDeep(nodes, iters[2 * 0 + 0], iters[2 * 0 + 1], level + 1);
        sub_indices[1] = CreateInterNodeDeep(nodes, iters[2 * 1 + 0], iters[2 * 1 + 1], level + 1);
        nodes[offset].nodes[0] = sub_indices[0];
        nodes[offset].nodes[1] = sub_indices[1];

        return offset;
    }
}

static inline void SplitCodeFace(code_face_iter iters_out[], int n, code_face_iter a, code_face_iter b, int level)
{
    if (n == 1)
    {
        iters_out[0] = a;
        iters_out[1] = b;
    }
    else
    {
        size_t sz = b - a;
        int n2 = n >> 1;
        code_face_iter c = b; //
        if (level < 3 * DIV_BIT)
        {
            c = std::partition(a, b, separator(level));
        }
        else
        {
            c = a + (sz >> 1);
        }
        code_face_iter iters[] = { a, c, c, b };
        {
#pragma omp parallel for
            for (int i = 0; i < 2; i++)
            {
                SplitCodeFace(iters_out + i * n, n2, iters[2 * i + 0], iters[2 * i + 1], level + 1);
            }
        }
    }
}

static inline size_t PushBranchNode(
    std::vector<InterNode>& nodes,
    size_t indices[], size_t sizes[], int n, int level)
{
    size_t offset = nodes.size();
    InterNode tmp;
    nodes.push_back(tmp);
    InterNode& node = nodes.back();

    node.dim = GetDim(level);
    node.pFace = NULL;
    //node.fsz = sz;

    if (n == 2)
    {
        nodes[offset].nodes[0] = indices[0];
        nodes[offset].nodes[1] = indices[1];
        nodes[offset].fsz = sizes[0] + sizes[1];
    }
    else
    {
        int n2 = n >> 1;
        size_t tmp[2] = {};
        tmp[0] = PushBranchNode(nodes, indices, sizes, n2, level + 1);
        tmp[1] = PushBranchNode(nodes, indices + n2, sizes + n2, n2, level + 1);

        nodes[offset].nodes[0] = tmp[0];
        nodes[offset].nodes[1] = tmp[1];
        nodes[offset].fsz = nodes[tmp[0]].fsz + nodes[tmp[1]].fsz;
    }
    return offset;
}

static inline size_t CreateInterNodeParallel(std::vector<InterNode>& nodes, code_face_iter a, code_face_iter b, int level, int para)
{
    size_t sz = b - a;
    if (sz == 0) return EMPTY_MASK;
    if (sz == 1)
    {
        size_t offset = nodes.size();
        InterNode node;
        node.dim = GetDim(level);
        node.pFace = a->pFace;
        node.fsz = 1;
        node.nodes[0] = EMPTY_MASK;
        node.nodes[1] = EMPTY_MASK;
        nodes.push_back(node);
        return offset;
    }
    else
    {
        size_t offset = nodes.size();

        std::vector<code_face_iter> iters(para * 2);
        SplitCodeFace(&iters[0], para, a, b, level);

        std::vector<size_t> sub_sizes(para);
        std::vector<size_t> sub_indices(para);
        std::vector<std::vector<InterNode> > sub_nodes(para);
        {
#pragma omp parallel for
            for (int i = 0; i < para; i++)
            {
                size_t fsz = sub_sizes[i] = iters[2 * i + 1] - iters[2 * i + 0];
                sub_nodes[i].reserve(3 * fsz);
                sub_indices[i] = CreateInterNodeDeep(sub_nodes[i], iters[2 * i + 0], iters[2 * i + 1], level + 1);
            }
        }

        std::vector<size_t> node_offsets(para + 1);
        for (int i = 0; i < para; i++)
        {
            node_offsets[i + 1] = 0;
        }

        node_offsets[0] = offset + 2 * para - 1;
        for (int i = 0; i < para; i++)
        {
            node_offsets[i + 1] = sub_nodes[i].size();
        }
        for (int i = 0; i < para; i++)
        {
            node_offsets[i + 1] += node_offsets[i];
        }

        for (int i = 0; i < para; i++)
        {
            size_t idx = sub_indices[i];
            if (IsBranch(idx))
            { //
                idx += node_offsets[i];
                sub_indices[i] = idx;
            }
        }

        nodes.reserve(node_offsets[para]);

        PushBranchNode(nodes, &sub_indices[0], &sub_sizes[0], para, level);

        nodes.resize(node_offsets[para]);

        {
#pragma omp parallel for
            for (int i = 0; i < para; i++)
            {
                std::vector<InterNode>& nx = sub_nodes[i];
                size_t jsz = nx.size();
                for (size_t j = 0; j < jsz; j++)
                {
                    InterNode& nd = nx[j];
                    for (int k = 0; k < 2; k++)
                    {
                        size_t idx = nd.nodes[k];
                        if (IsBranch(idx))
                        { //
                            idx += node_offsets[i];
                            nd.nodes[k] = idx;
                        }
                    }
                }
                if (!sub_nodes[i].empty())
                {
                    memcpy(&nodes[node_offsets[i]], &(sub_nodes[i][0]), sizeof(InterNode) * sub_nodes[i].size());
                }
            }
        }

        return offset;
    }
}

static inline size_t CreateInterNode(std::vector<InterNode>& nodes, code_face_iter a, code_face_iter b, int level, int para)
{
    if (para <= 1)
        return CreateInterNodeDeep(nodes, a, b, level);
    else
        return CreateInterNodeParallel(nodes, a, b, level, para);
}

static inline void ExpandIndices(size_t indices[], int n, const std::vector<InterNode>& nodes, size_t index)
{
    if (index == EMPTY_MASK) return;
    const InterNode& node = nodes[index];
    if (node.pFace)
    {
        indices[0] = index;
    }
    else
    {
        if (n == 4)
        {

            if (node.nodes[0] != EMPTY_MASK)
            {
                const InterNode& cnode = nodes[node.nodes[0]];
                if (cnode.pFace)
                {
                    indices[0] = node.nodes[0];
                    indices[1] = EMPTY_MASK;
                }
                else
                {
                    indices[0] = cnode.nodes[0];
                    indices[1] = cnode.nodes[1];
                }
            }
            if (node.nodes[1] != EMPTY_MASK)
            {
                const InterNode& cnode = nodes[node.nodes[1]];
                if (cnode.pFace)
                {
                    indices[2] = node.nodes[1];
                    indices[3] = EMPTY_MASK;
                }
                else
                {
                    indices[2] = cnode.nodes[0];
                    indices[3] = cnode.nodes[1];
                }
            }
        }
        else
        {
            int n2 = n >> 1;
            ExpandIndices(indices, n2, nodes, node.nodes[0]);
            ExpandIndices(indices + n2, n2, nodes, node.nodes[1]);
        }
    }
}

static inline int CompactionIndices(size_t indices[], int n)
{
    int nRet = 0;
    for (int i = 0; i < n; i++)
    {
        if (indices[i] != EMPTY_MASK)
        {
            indices[nRet] = indices[i];
            nRet++;
        }
    }
    for (int i = nRet; i < n; i++)
    {
        indices[i] = EMPTY_MASK;
    }
    return nRet;
}

static inline int GetSubNodeSize4(int para)
{
    switch (para)
    {
    case 4:
        return 1; //1
    case 16:
        return 5; //4+1
    case 64:
        return 21; //16+4+1
    }
    int sum = 0;
    int end = para / 4;
    int x = 1;
    while (x <= end)
    {
        sum += x;
        x *= 4;
    }
    return sum;
}

struct CandidateNode
{
    bool bReplace; //
    size_t index;  //Self Index
    size_t fsz;    //face size
    int dim;
    float area;      //
    vec3 min;    //area;
    vec3 max;    //area;
    size_t nodes[2]; //
};
typedef std::vector<CandidateNode> CandidateTree;

static inline uint32_t GetExistMask(size_t indeces[], int n)
{
    uint32_t nRet = 0;
    for (int i = 0; i < n; i++)
    {
        if (indeces[i] != EMPTY_MASK)
        {
            nRet |= 1 << i;
        }
    }
    return nRet;
}

static const int popCountArray[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8 };

static inline int PopCount(uint32_t p)
{
    assert(0x0 <= p && p <= 0xFF);
    return popCountArray[p];
}

static const int selectArray[] = {
    0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
};

static inline int Select(uint32_t p)
{
    assert(0x0 <= p && p <= 0xFF);
    return selectArray[p]; //Find First index
}

static inline float GetLeafCost(float area, size_t sz)
{
    return Ct() * area * sz;
}

static inline float GetBranchCost(float area, float Cl, float Cr)
{
    return Ci() * area + Cl + Cr;
}

//------------------------------------------------------------------------------------------------------------------

template <int N>
static inline void GetMinMax(vec3& min, vec3& max, const CandidateNode nodes[], uint32_t nMask)
{
    static const float far_ = std::numeric_limits<float>::max();

    min = vec3(+far_, +far_, +far_);
    max = vec3(-far_, -far_, -far_);
    for (int i = 0; i < N; i++)
    {
        if ((nMask & (1 << i)) && nodes[i].index != EMPTY_MASK)
        {
            for (int j = 0; j < 3; j++)
            {
                min[j] = std::min(min[j], nodes[i].min[j]);
                max[j] = std::max(max[j], nodes[i].max[j]);
            }
        }
    }
}

inline size_t GetTotalTriangles(const CandidateNode nodes[], int n, uint32_t nMask)
{
    size_t sz = 0;
    for (int i = 0; i < n; i++)
    {
        if ((nMask & (1 << i)) && nodes[i].index != EMPTY_MASK)
        {
            sz += nodes[i].fsz;
        }
    }
    return sz;
}

static inline void CreateInternalCandidateTree(CandidateTree& tree, const vec3& min, const vec3& max, float area, size_t fsz, CandidateTree& ltree, CandidateTree& rtree)
{
    size_t lsz = ltree.size();
    size_t rsz = rtree.size();
    size_t li = 1;
    size_t ri = 1 + lsz;
    CandidateNode node;
    node.index = 0;
    node.fsz = fsz;
    node.area = area;
    node.bReplace = true;
    node.nodes[0] = li;
    node.nodes[1] = ri;
    node.min = min;
    node.max = max;

    tree.reserve(1 + lsz + rsz);
    tree.push_back(node);
    for (size_t i = 0; i < lsz; i++)
    {
        CandidateNode& n = ltree[i];
        if (n.bReplace)
        {
            if (n.nodes[0] != EMPTY_MASK) n.nodes[0] += li;
            if (n.nodes[1] != EMPTY_MASK) n.nodes[1] += li;
        }
        tree.push_back(n);
    }
    for (size_t i = 0; i < rsz; i++)
    {
        CandidateNode& n = rtree[i];
        if (n.bReplace)
        {
            if (n.nodes[0] != EMPTY_MASK) n.nodes[0] += ri;
            if (n.nodes[1] != EMPTY_MASK) n.nodes[1] += ri;
        }
        tree.push_back(n);
    }
}

static inline void GetMinMax(vec3& min, vec3& max, const CandidateNode nodes[], int n, int nMask)
{
    //---------------------------------------------------------------------
    static const float far_ = std::numeric_limits<float>::max();
    //---------------------------------------------------------------------
    min = vec3(+far_, +far_, +far_);
    max = vec3(-far_, -far_, -far_);
    for (int i = 0; i < n; i++)
    {
        if ((1 << i) & nMask)
        {
            vec3 cmin = nodes[i].min;
            vec3 cmax = nodes[i].max;
            for (int j = 0; j < 3; j++)
            {
                min[j] = std::min(min[j], cmin[j]);
                max[j] = std::max(max[j], cmax[j]);
            }
        }
    }
}

static inline void GetBoundsTreelet(vec3& min, vec3& max, CandidateTree& tree, size_t idx)
{
    //---------------------------------------------------------------------
    static const float far_ = std::numeric_limits<float>::max();
    //---------------------------------------------------------------------
    if (idx == EMPTY_MASK)
    {
        min = vec3(+far_, +far_, +far_);
        max = vec3(-far_, -far_, -far_);
        return;
    }
    if (!tree[idx].bReplace)
    {
        min = tree[idx].min;
        max = tree[idx].max;
        return;
    }
    else
    {
        vec3 mins[2];
        vec3 maxs[2];
        for (int i = 0; i < 2; i++)
        {
            GetBoundsTreelet(mins[i], maxs[i], tree, tree[idx].nodes[i]);
        }
        min = mins[0];
        max = maxs[0];
        for (int j = 0; j < 3; j++)
        {
            min[j] = std::min(min[j], mins[1][j]);
            max[j] = std::max(max[j], maxs[1][j]);
        }

        int dim = 0;

        int nNodes = 0;
        if (tree[idx].nodes[0] != EMPTY_MASK) nNodes++;
        if (tree[idx].nodes[1] != EMPTY_MASK) nNodes++;
        if (nNodes == 1)
        {
            vec3 wid = max - min;
            if (wid[1] > wid[dim]) dim = 1;
            if (wid[2] > wid[dim]) dim = 2;

            if (tree[idx].nodes[0] == EMPTY_MASK)
            {
                std::swap(tree[idx].nodes[0], tree[idx].nodes[1]);
            }
        }
        else if (nNodes == 2)
        {
            vec3 c[2];
            for (int i = 0; i < 2; i++)
            {
                c[i] = (maxs[i] + mins[i]) * 0.5f;
            }
            vec3 wid = c[1] - c[0];
            for (int i = 0; i < 3; i++)
                wid[i] = fabs(wid[i]);
            dim = 0;
            if (wid[1] > wid[dim]) dim = 1;
            if (wid[2] > wid[dim]) dim = 2;
            if (c[0][dim] > c[1][dim])
            {
                std::swap(tree[idx].nodes[0], tree[idx].nodes[1]);
            }
        }

        tree[idx].min = min;
        tree[idx].max = max;
        tree[idx].area = GetArea(min, max);
        tree[idx].dim = dim;
    }
}

static inline size_t GetOptimalTreelet(CandidateTree& tree, const CandidateNode nodes[], int P[], int i)
{
    int s = i;
    int sz = PopCount(s);

    if (sz == 0) return EMPTY_MASK;
    if (sz == 1)
    {
        size_t offset = tree.size();
        int idx = Select(i);
        tree.push_back(nodes[idx]);
        return offset;
    }
    else
    {
        size_t offset = tree.size();
        CandidateNode node;
        node.bReplace = true;
        node.index = offset;
        tree.push_back(node);

        int p = P[i];

        size_t indices[2];
        indices[0] = GetOptimalTreelet(tree, nodes, P, p);
        indices[1] = GetOptimalTreelet(tree, nodes, P, s ^ p);

        tree[offset].nodes[0] = indices[0];
        tree[offset].nodes[1] = indices[1];
        tree[offset].fsz =
            ((indices[0] == EMPTY_MASK) ? 0 : tree[indices[0]].fsz) +
            ((indices[1] == EMPTY_MASK) ? 0 : tree[indices[1]].fsz);

        return offset;
    }
}

template <int N>
static inline void GetOptimalTreelet(CandidateTree& tree, const CandidateNode nodes[], int n)
{
    //---------------------------------------------------------------------
    static const float far_ = std::numeric_limits<float>::max();
    //---------------------------------------------------------------------

    if (n == 1)
    {
        tree.push_back(nodes[0]);
    }
    else if (n == 2)
    {
        tree.reserve(2 * 2 - 1);
        size_t offset = tree.size();
        CandidateNode node;
        node.bReplace = true;
        node.index = 0;
        node.fsz = nodes[0].fsz + nodes[1].fsz;
        tree.push_back(node);
        tree.push_back(nodes[0]);
        tree.push_back(nodes[1]);
        tree[offset].nodes[0] = 1;
        tree[offset].nodes[1] = 2;
    }
    else
    {
        //Calculate surface area for each subset
        int nTotal = (1 << n); //2^n
        //std::vector<float> areas(nTotal);
        float areas[1 << N];
        areas[0] = 0;
        for (int i = 1; i < nTotal; i++) //1..255
        {
            vec3 min = vec3(+far_, +far_, +far_);
            vec3 max = vec3(-far_, -far_, -far_);
            GetMinMax(min, max, nodes, n, i);
            areas[i] = GetArea(min, max);
        }
        // Initialize costs of individual leaves
        //std::vector<float> Copt(nTotal);
        float Copt[1 << N];
        for (int i = 0; i < nTotal; i++) //1..255
        {
            Copt[i] = far_;
        }
        for (int i = 0; i < n; i++)
        {
            int idx = 1 << i;
            Copt[idx] = GetLeafCost(nodes[i].area, nodes[i].fsz);
        }
        //std::vector<int> Popt(nTotal);
        int Popt[1 << N];
        memset(&Popt[0], 0, sizeof(int) * nTotal);

        for (int k = 2; k <= n; k++)
        {
            int kTotal = 1 << k;
            for (int i = 1; i < kTotal; i++)
            {
                float Cs = far_;
                int Ps = 0;

                int s = i;
                int d = (s - 1) & s;
                int p = (-d) & s;

                do
                {
                    float c = Copt[p] + Copt[s ^ p];
                    if (c < Cs)
                    {
                        Cs = c;
                        Ps = p;
                    }
                    p = (p - d) & s;
                } while (p != 0);
                size_t t = GetTotalTriangles(nodes, n, s);
                Copt[s] = std::min<float>(Ci() * areas[s] + Cs, Ct() * areas[s] * t);
                Popt[s] = Ps;
            }
        }

        //float totalCost = Copt[(1<<n)-1];
        tree.reserve(2 * n - 1);
        GetOptimalTreelet(tree, nodes, &Popt[0], (1 << n) - 1);
    }

    {
        vec3 tmin, tmax;
        GetBoundsTreelet(tmin, tmax, tree, 0);
    }
}

static inline void ExpandIndices2(size_t indices[], int n, std::vector<size_t>& internals, const std::vector<InterNode>& nodes, size_t index)
{
    if (index == EMPTY_MASK) return;
    const InterNode& node = nodes[index];
    if (node.pFace)
    {
        indices[0] = index;
    }
    else
    {
        internals.push_back(index);

        if (n == 4)
        {

            if (node.nodes[0] != EMPTY_MASK)
            {
                const InterNode& cnode = nodes[node.nodes[0]];
                if (cnode.pFace)
                {
                    indices[0] = node.nodes[0];
                    indices[1] = EMPTY_MASK;
                }
                else
                {
                    internals.push_back(node.nodes[0]);

                    indices[0] = cnode.nodes[0];
                    indices[1] = cnode.nodes[1];
                }
            }
            if (node.nodes[1] != EMPTY_MASK)
            {
                const InterNode& cnode = nodes[node.nodes[1]];
                if (cnode.pFace)
                {
                    indices[2] = node.nodes[1];
                    indices[3] = EMPTY_MASK;
                }
                else
                {
                    internals.push_back(node.nodes[1]);

                    indices[2] = cnode.nodes[0];
                    indices[3] = cnode.nodes[1];
                }
            }
        }
        else
        {
            int n2 = n >> 1;
            ExpandIndices2(indices, n2, internals, nodes, node.nodes[0]);
            ExpandIndices2(indices + n2, n2, internals, nodes, node.nodes[1]);
        }
    }
}

static inline size_t ReplaceTreelet(std::vector<size_t>& internals, const CandidateTree& tree, size_t idx, std::vector<InterNode>& nodes)
{
    if (idx == EMPTY_MASK) return EMPTY_MASK;
    if (!tree[idx].bReplace) return tree[idx].index;
    size_t index = internals.back();
    internals.pop_back();
    nodes[index].fsz = tree[idx].fsz;
    nodes[index].dim = tree[idx].dim;
    nodes[index].nodes[0] = ReplaceTreelet(internals, tree, tree[idx].nodes[0], nodes);
    nodes[index].nodes[1] = ReplaceTreelet(internals, tree, tree[idx].nodes[1], nodes);
    return index;
}

template <int N>
static inline void OptimizeInterNodeDeep(vec3& min, vec3& max, std::vector<InterNode>& nodes, size_t index)
{
    //---------------------------------------------------------------------
    static const float far_ = std::numeric_limits<float>::max();
    //---------------------------------------------------------------------

    if (index == EMPTY_MASK) return;
    assert(index < nodes.size());
    if (nodes[index].pFace) //Is Leaf
    {
        min = vec3(+far_, +far_, +far_);
        max = vec3(-far_, -far_, -far_);
        GetMinMax(min, max, nodes[index].pFace);
    }
    else
    {
        size_t indices[N];
        for (int i = 0; i < N; i++)
        { //
            indices[i] = EMPTY_MASK;
        }

        std::vector<size_t> internals;
        internals.reserve(N - 1);

        ExpandIndices2(indices, N, internals, nodes, index);
        int n = CompactionIndices(indices, N);

        vec3 mins[N];
        vec3 maxs[N];
        for (int i = 0; i < n; i++)
        {
            mins[i] = vec3(+far_, +far_, +far_);
            maxs[i] = vec3(-far_, -far_, -far_);
        }
        //---------------------------------------------------------------------
        for (int i = 0; i < n; i++)
        { //
            OptimizeInterNodeDeep<N>(mins[i], maxs[i], nodes, indices[i]);
        }
        //---------------------------------------------------------------------
        if (n == 1)
        {
            min = mins[0];
            max = maxs[0];
        }
        else
        {
            //---------------------------------------------------------------------
            CandidateNode cNodes[N];
            for (int i = 0; i < n; i++)
            { //
                cNodes[i].index = indices[i];
                cNodes[i].area = GetArea(mins[i], maxs[i]);
                cNodes[i].min = mins[i];
                cNodes[i].max = maxs[i];
                cNodes[i].bReplace = false; //fixed node
                if (indices[i] != EMPTY_MASK)
                {
                    cNodes[i].fsz = nodes[indices[i]].fsz;
                }
                else
                {
                    cNodes[i].fsz = 0;
                }
            }
            //---------------------------------------------------------------------
            CandidateTree tree;
            GetOptimalTreelet<N>(tree, cNodes, n);
            //---------------------------------------------------------------------
            //std::sort(internals.begin(), internals.end(), std::greater<size_t>());
            std::reverse(internals.begin(), internals.end());
            assert(index == internals.back());
            ReplaceTreelet(internals, tree, 0, nodes);

            //---------------------------------------------------------------------
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    min[j] = std::min(min[j], mins[i][j]);
                    max[j] = std::max(max[j], maxs[i][j]);
                }
            }
        }
    }
}

template <int N>
static inline void OptimizeInterNodeParallel(vec3& min, vec3& max, std::vector<InterNode>& nodes, size_t index, int para)
{
    //---------------------------------------------------------------------
    static const float far_ = std::numeric_limits<float>::max();
    //---------------------------------------------------------------------

    if (index == EMPTY_MASK) return;
    assert(index < nodes.size());
    if (nodes[index].pFace) //Is Leaf
    {
        min = vec3(+far_, +far_, +far_);
        max = vec3(-far_, -far_, -far_);
        GetMinMax(min, max, nodes[index].pFace);
    }
    else
    {
        size_t indices[N];
        for (int i = 0; i < N; i++)
        { //
            indices[i] = EMPTY_MASK;
        }

        std::vector<size_t> internals;
        internals.reserve(N - 1);

        ExpandIndices2(indices, N, internals, nodes, index);
        int n = CompactionIndices(indices, N);

        vec3 mins[N];
        vec3 maxs[N];
        for (int i = 0; i < n; i++)
        {
            mins[i] = vec3(+far_, +far_, +far_);
            maxs[i] = vec3(-far_, -far_, -far_);
        }
        //---------------------------------------------------------------------
        {
#pragma omp parallel for
            for (int i = 0; i < n; i++)
            { //
                OptimizeInterNodeDeep<N>(mins[i], maxs[i], nodes, indices[i]);
            }
        }
        //---------------------------------------------------------------------
        if (n == 1)
        {
            min = mins[0];
            max = maxs[0];
        }
        else
        {
            //---------------------------------------------------------------------
            CandidateNode cNodes[N];
            for (int i = 0; i < n; i++)
            { //
                cNodes[i].index = indices[i];
                cNodes[i].area = GetArea(mins[i], maxs[i]);
                cNodes[i].min = mins[i];
                cNodes[i].max = maxs[i];
                cNodes[i].bReplace = false; //fixed node
                if (indices[i] != EMPTY_MASK)
                {
                    cNodes[i].fsz = nodes[indices[i]].fsz;
                }
                else
                {
                    cNodes[i].fsz = 0;
                }
            }
            //---------------------------------------------------------------------
            CandidateTree tree;
            GetOptimalTreelet<N>(tree, cNodes, n);
            //---------------------------------------------------------------------
            //std::sort(internals.begin(), internals.end(), std::greater<size_t>());
            std::reverse(internals.begin(), internals.end());
            assert(index == internals.back());
            ReplaceTreelet(internals, tree, 0, nodes);

            //InterNode node_ = nodes[index];

            //---------------------------------------------------------------------
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    min[j] = std::min(min[j], mins[i][j]);
                    max[j] = std::max(max[j], maxs[i][j]);
                }
            }
        }
    }
}

static inline void OptimizeInterNode(std::vector<InterNode>& nodes, size_t index, int para)
{
    static const int N = 8;
    vec3 min;
    vec3 max;
    if (para < 2)
        OptimizeInterNodeDeep<N>(min, max, nodes, index); //TODO
    else
        OptimizeInterNodeParallel<N>(min, max, nodes, index, para); //TODO
}

static inline float GetCostSAH(vec3& min, vec3& max, const std::vector<InterNode>& nodes, size_t index)
{
    static const float far_ = std::numeric_limits<float>::max();

    if (index == EMPTY_MASK)
    {
        min = vec3(+far_, +far_, +far_);
        max = vec3(-far_, -far_, -far_);
        return 0;
    }
    const InterNode& node = nodes[index];
    if (node.pFace)
    {
        min = vec3(+far_, +far_, +far_);
        max = vec3(-far_, -far_, -far_);
        GetMinMax(min, max, node.pFace);
        float area = GetArea(min, max);
        return GetLeafCost(area, 1);
    }
    else
    {
        vec3 mins[2];
        vec3 maxs[2];
        float Cl = GetCostSAH(mins[0], maxs[0], nodes, node.nodes[0]);
        float Cr = GetCostSAH(mins[1], maxs[1], nodes, node.nodes[1]);

        min = mins[0];
        max = maxs[0];
        for (int j = 0; j < 3; j++)
        {
            min[j] = std::min(min[j], mins[1][j]);
            max[j] = std::max(max[j], maxs[1][j]);
        }

        float area = GetArea(min, max);
        return std::min<float>(GetLeafCost(area, node.fsz), GetBranchCost(area, Cl, Cr));
    }
}

static inline float GetCostSAH(const std::vector<InterNode>& nodes)
{
    vec3 min, max;
    return GetCostSAH(min, max, nodes, 0);
}

static inline size_t get_branch_node_size(size_t face_num)
{
    if (face_num <= MIN_FACE) return 1;
    size_t p = face_num / 4;
    if (face_num & 3) p++;
    return std::max<size_t>(1, 1 + 4 * get_branch_node_size(p));
}
static inline size_t get_leaf_node_size(size_t face_num)
{
    return std::max<size_t>(MIN_FACE, face_num + (int)ceil(double(face_num) / (MIN_FACE)));
}

static inline void CheckBound(vec3& min, vec3& max)
{
    static const float EPS = std::numeric_limits<float>::epsilon() * 256;

    vec3 wid = (max - min) * 0.5f;
    vec3 cnt = (max + min) * 0.5f;
    for (int i = 0; i < 3; i++)
    {
        wid[i] = std::max<float>(wid[i], EPS);
    }

    min = cnt - wid;
    max = cnt + wid;
}
//------------------------------------------------------------------------------------------------------------------

static inline size_t ConvertInterNodeToQBVHDeep(
    std::vector<PCFACE>& out_faces, aligned_vector<SIMDBVHNode>& out_nodes,
    vec3& min, vec3& max,
    const std::vector<InterNode>& nodes, size_t index,
    int level)
{
    static const float EPS = std::numeric_limits<float>::epsilon();
    static const float far_ = std::numeric_limits<float>::max();
    static const int TRAP[] = { 0, 1, 2, 0, 1, 2 };

    if (index == EMPTY_MASK) return EMPTY_MASK;
    //if(nodes.size()<=index)return EMPTY_MASK;
    const InterNode& node = nodes[index];
    size_t sz = GetFaceSize(nodes, index);
    if (sz == 0) return EMPTY_MASK;
    if (level != 0 && sz <= MIN_FACE || level >= DIV_BIT * 3)
    { //
        std::vector<PCFACE> faces;
        faces.reserve(sz);
        GetFaces(faces, nodes, index);
        assert(faces.size() == sz);
        size_t first = out_faces.size();
        size_t last = first + sz + 1; //zero terminate
        out_faces.resize(last);
        memcpy(&out_faces[first], &faces[0], sizeof(PCFACE) * sz);
        out_faces[last - 1] = 0;
        vec3 cmin(+far_, +far_, +far_);
        vec3 cmax(-far_, -far_, -far_);
        GetMinMax(cmin, cmax, faces);

        min = cmin - vec3(EPS, EPS, EPS);
        max = cmax + vec3(EPS, EPS, EPS);

        CheckBound(min, max);

        size_t nRet = MakeLeafIndex(first);
        assert(IsLeaf(nRet));
        return nRet;
    }
    else
    {
        size_t offset = out_nodes.size();

        int dim = node.dim;
        int dim1 = TRAP[dim + 1];
        SIMDBVHNode snode;
        snode.axis_top = dim;
        snode.axis_left = dim1;
        snode.axis_right = dim1;

        size_t indices[4] = { EMPTY_MASK, EMPTY_MASK, EMPTY_MASK, EMPTY_MASK };

        if (node.nodes[0] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[0]];
            snode.axis_left = cnode.dim;
            if (cnode.pFace)
            {
                indices[0] = node.nodes[0];
                indices[1] = EMPTY_MASK;
            }
            else
            {
                indices[0] = cnode.nodes[0];
                indices[1] = cnode.nodes[1];
            }
        }
        if (node.nodes[1] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[1]];
            snode.axis_right = cnode.dim;
            if (cnode.pFace)
            {
                indices[2] = node.nodes[1];
                indices[3] = EMPTY_MASK;
            }
            else
            {
                indices[2] = cnode.nodes[0];
                indices[3] = cnode.nodes[1];
            }
        }

        out_nodes.push_back(snode);

        vec3 minmax[4][2];
        for (int i = 0; i < 4; i++)
        {
            minmax[i][0] = vec3(+far_, +far_, +far_);
            minmax[i][1] = vec3(-far_, -far_, -far_);
        }

        {
            size_t sub_indices[4] = {};
            sub_indices[0] = ConvertInterNodeToQBVHDeep(out_faces, out_nodes, minmax[0][0], minmax[0][1], nodes, indices[0], level + 2);
            sub_indices[1] = ConvertInterNodeToQBVHDeep(out_faces, out_nodes, minmax[1][0], minmax[1][1], nodes, indices[1], level + 2);
            sub_indices[2] = ConvertInterNodeToQBVHDeep(out_faces, out_nodes, minmax[2][0], minmax[2][1], nodes, indices[2], level + 2);
            sub_indices[3] = ConvertInterNodeToQBVHDeep(out_faces, out_nodes, minmax[3][0], minmax[3][1], nodes, indices[3], level + 2);
            out_nodes[offset].children[0] = sub_indices[0];
            out_nodes[offset].children[1] = sub_indices[1];
            out_nodes[offset].children[2] = sub_indices[2];
            out_nodes[offset].children[3] = sub_indices[3];
        }

        //convert & swizzle
        float bboxes[2][3][4];
        //for(int m = 0;m<2;m++){//minmax
        for (int j = 0; j < 3; j++)
        { //xyz
            for (int k = 0; k < 4; k++)
            {                                      //box
                bboxes[0][j][k] = minmax[k][0][j]; //
                bboxes[1][j][k] = minmax[k][1][j]; //
            }
        }
        //}

        //for(int i = 0;i<4;i++){
        for (int m = 0; m < 2; m++)
        { //minmax
            for (int j = 0; j < 3; j++)
            { //xyz
                out_nodes[offset].bboxes[m][j] = _mm_setzero_ps();
                out_nodes[offset].bboxes[m][j] = _mm_loadu_ps(bboxes[m][j]);
            }
        }
        //}

        min = minmax[0][0];
        max = minmax[0][1];
        for (int i = 1; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            { //xyz
                min[j] = std::min(min[j], minmax[i][0][j]);
                max[j] = std::max(max[j], minmax[i][1][j]);
            }
        }

        min -= vec3(EPS, EPS, EPS);
        max += vec3(EPS, EPS, EPS);

        return offset;
    }
}

static inline size_t PushBranchNode(
    aligned_vector<SIMDBVHNode>& out_nodes,
    vec3& min, vec3& max,
    const std::vector<InterNode>& nodes, size_t index, size_t sub_indices[], vec3 mins[], vec3 maxs[], int n, int level)
{
    static const float EPS = std::numeric_limits<float>::epsilon();
    static const float far_ = std::numeric_limits<float>::max();
    static const int TRAP[] = { 0, 1, 2, 0, 1, 2 };

    size_t offset = out_nodes.size();

    int dim = GetDim(level);
    if (index != EMPTY_MASK)
    {
        dim = nodes[index].dim;
    }
    int dim1 = TRAP[dim + 1];
    SIMDBVHNode snode;
    snode.axis_top = dim;
    snode.axis_left = dim1;
    snode.axis_right = dim1;

    size_t indices[4] = { EMPTY_MASK, EMPTY_MASK, EMPTY_MASK, EMPTY_MASK };

    if (index != EMPTY_MASK)
    {
        const InterNode& node = nodes[index];
        if (node.nodes[0] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[0]];
            snode.axis_left = cnode.dim;
            if (cnode.pFace)
            {
                indices[0] = node.nodes[0];
                indices[1] = EMPTY_MASK;
            }
            else
            {
                indices[0] = cnode.nodes[0];
                indices[1] = cnode.nodes[1];
            }
        }
        if (node.nodes[1] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[1]];
            snode.axis_right = cnode.dim;
            if (cnode.pFace)
            {
                indices[2] = node.nodes[1];
                indices[3] = EMPTY_MASK;
            }
            else
            {
                indices[2] = cnode.nodes[0];
                indices[3] = cnode.nodes[1];
            }
        }
    }
    out_nodes.push_back(snode);

    if (n == 4)
    {
        out_nodes[offset].children[0] = sub_indices[0];
        out_nodes[offset].children[1] = sub_indices[1];
        out_nodes[offset].children[2] = sub_indices[2];
        out_nodes[offset].children[3] = sub_indices[3];

        //convert & swizzle
        float bboxes[2][3][4];
        //for(int m = 0;m<2;m++){//minmax
        for (int j = 0; j < 3; j++)
        { //xyz
            for (int k = 0; k < 4; k++)
            {                                 //box
                bboxes[0][j][k] = mins[k][j]; //
                bboxes[1][j][k] = maxs[k][j]; //
            }
        }
        //}

        //for(int i = 0;i<4;i++){
        for (int m = 0; m < 2; m++)
        { //minmax
            for (int j = 0; j < 3; j++)
            { //xyz
                out_nodes[offset].bboxes[m][j] = _mm_setzero_ps();
                out_nodes[offset].bboxes[m][j] = _mm_loadu_ps(bboxes[m][j]);
            }
        }
        //}
    }
    else
    {
        int n2 = n >> 2;
        size_t tmp[4] = {};
        vec3 cmin[4];
        vec3 cmax[4];

        tmp[0] = PushBranchNode(out_nodes, cmin[0], cmax[0], nodes, indices[0], sub_indices + 0 * n2, mins + 0 * n2, maxs + 0 * n2, n2, level + 2);
        tmp[1] = PushBranchNode(out_nodes, cmin[1], cmax[1], nodes, indices[1], sub_indices + 1 * n2, mins + 1 * n2, maxs + 1 * n2, n2, level + 2);
        tmp[2] = PushBranchNode(out_nodes, cmin[2], cmax[2], nodes, indices[2], sub_indices + 2 * n2, mins + 2 * n2, maxs + 2 * n2, n2, level + 2);
        tmp[3] = PushBranchNode(out_nodes, cmin[3], cmax[3], nodes, indices[3], sub_indices + 3 * n2, mins + 3 * n2, maxs + 3 * n2, n2, level + 2);

        out_nodes[offset].children[0] = tmp[0];
        out_nodes[offset].children[1] = tmp[1];
        out_nodes[offset].children[2] = tmp[2];
        out_nodes[offset].children[3] = tmp[3];

        //convert & swizzle
        float bboxes[2][3][4];
        //for(int m = 0;m<2;m++){//minmax
        for (int j = 0; j < 3; j++)
        { //xyz
            for (int k = 0; k < 4; k++)
            {                                 //box
                bboxes[0][j][k] = cmin[k][j]; //
                bboxes[1][j][k] = cmax[k][j]; //
            }
        }
        //}

        //for(int i = 0;i<4;i++){
        for (int m = 0; m < 2; m++)
        { //minmax
            for (int j = 0; j < 3; j++)
            { //xyz
                out_nodes[offset].bboxes[m][j] = _mm_setzero_ps();
                out_nodes[offset].bboxes[m][j] = _mm_loadu_ps(bboxes[m][j]);
            }
        }
        //}

        min = cmin[0];
        max = cmax[0];
        for (int i = 1; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            { //xyz
                if (min[j] > cmin[i][j]) min[j] = cmin[i][j];
                if (max[j] < cmax[i][j]) max[j] = cmax[i][j];
            }
        }

        min -= vec3(EPS, EPS, EPS);
        max += vec3(EPS, EPS, EPS);
    }

    return offset;
}

static inline size_t ConvertInterNodeToQBVHParallel(
    std::vector<PCFACE>& out_faces, aligned_vector<SIMDBVHNode>& out_nodes,
    vec3& min, vec3& max,
    const std::vector<InterNode>& nodes, size_t index,
    int level, int para)
{
    static const float EPS = std::numeric_limits<float>::epsilon();
    static const float far_ = std::numeric_limits<float>::max();
    static const int TRAP[] = { 0, 1, 2, 0, 1, 2 };

    if (index == EMPTY_MASK) return EMPTY_MASK;
    //if(nodes.size()<=index)return EMPTY_MASK;

    size_t sz = GetFaceSize(nodes, index);
    if (sz == 0) return EMPTY_MASK;
    if (level != 0 && sz <= MIN_FACE || level >= DIV_BIT * 3)
    { //
        std::vector<PCFACE> faces;
        faces.reserve(sz);
        GetFaces(faces, nodes, index);
        assert(faces.size() == sz);
        size_t first = out_faces.size();
        size_t last = first + sz + 1; //zero terminate
        out_faces.resize(last);
        memcpy(&out_faces[first], &faces[0], sizeof(PCFACE) * sz);
        out_faces[last - 1] = 0;
        vec3 cmin(+far_, +far_, +far_);
        vec3 cmax(-far_, -far_, -far_);
        GetMinMax(cmin, cmax, faces);

        min = cmin - vec3(EPS, EPS, EPS);
        max = cmax + vec3(EPS, EPS, EPS);

        CheckBound(min, max);

        size_t nRet = MakeLeafIndex(first);
        assert(IsLeaf(nRet));
        return nRet;
    }
    else
    {
        size_t offset = out_nodes.size();

        std::vector<size_t> indices(para);
        for (size_t i = 0; i < para; i++)
        {
            indices[i] = EMPTY_MASK;
        }

        ExpandIndices(&indices[0], para, nodes, index);

        std::vector<vec3> cmin(para);
        std::vector<vec3> cmax(para);
        for (int i = 0; i < para; i++)
        {
            cmin[i] = vec3(+far_, +far_, +far_);
            cmax[i] = vec3(-far_, -far_, -far_);
        }

        {
            std::vector<size_t> sub_indices(para);
            std::vector<std::vector<PCFACE> > sub_faces(para);
            std::vector<aligned_vector<SIMDBVHNode> > sub_nodes(para);
            {
#pragma omp parallel for
                for (int i = 0; i < para; i++)
                {
                    size_t fsz = GetFaceSize(nodes, indices[i]);
                    sub_faces[i].reserve(get_leaf_node_size(fsz));
                    sub_nodes[i].reserve(get_branch_node_size(fsz));
                    sub_indices[i] = ConvertInterNodeToQBVHDeep(sub_faces[i], sub_nodes[i], cmin[i], cmax[i], nodes, indices[i], level + 2);
                }
            }

            std::vector<size_t> face_offsets(para + 1);
            std::vector<size_t> node_offsets(para + 1);
            face_offsets[0] = 0;
            node_offsets[0] = GetSubNodeSize4(para); //(para/4):4->1, 16->4+1, 64->16+4+1=21
            for (int i = 0; i < para; i++)
            {
                face_offsets[i + 1] = sub_faces[i].size();
                node_offsets[i + 1] = sub_nodes[i].size();
            }
            for (int i = 0; i < para; i++)
            {
                face_offsets[i + 1] += face_offsets[i];
                node_offsets[i + 1] += node_offsets[i];
            }

            for (int i = 0; i < para; i++)
            {
                size_t idx = sub_indices[i];
                if (IsBranch(idx))
                { //
                    idx += node_offsets[i];
                    sub_indices[i] = idx;
                }
                else if (!IsEmpty(idx))
                { //
                    idx = GetFaceFirst(idx);
                    idx += face_offsets[i];
                    sub_indices[i] = MakeLeafIndex(idx);
                }
            }

            out_faces.reserve(face_offsets[para]);
            out_nodes.reserve(node_offsets[para]);

            PushBranchNode(out_nodes, min, max, nodes, index, &sub_indices[0], &cmin[0], &cmax[0], para, level);

            out_faces.resize(face_offsets[para]);
            out_nodes.resize(node_offsets[para]);

            {
#pragma omp parallel for
                for (int i = 0; i < para; i++)
                {
                    aligned_vector<SIMDBVHNode>& nodes = sub_nodes[i];
                    size_t jsz = nodes.size();
                    for (size_t j = 0; j < jsz; j++)
                    {
                        SIMDBVHNode& nd = nodes[j];
                        for (int k = 0; k < 4; k++)
                        {
                            size_t idx = nd.children[k];
                            if (IsBranch(idx))
                            { //
                                idx += node_offsets[i];
                                nd.children[k] = idx;
                            }
                            else if (!IsEmpty(idx))
                            { //
                                idx = GetFaceFirst(idx);
                                idx += face_offsets[i];
                                nd.children[k] = MakeLeafIndex(idx);
                            }
                        }
                    }

                    if (!sub_faces[i].empty())
                    {
                        memcpy(&out_faces[face_offsets[i]], &(sub_faces[i][0]), sizeof(PCFACE) * sub_faces[i].size());
                    }
                    if (!sub_nodes[i].empty())
                    {
                        memcpy(&out_nodes[node_offsets[i]], &(sub_nodes[i][0]), sizeof(SIMDBVHNode) * sub_nodes[i].size());
                    }
                }
            }
        }

        return offset;
    }
}

static inline size_t ConvertInterNodeToQBVH(
    std::vector<PCFACE>& out_faces, aligned_vector<SIMDBVHNode>& out_nodes,
    vec3& min, vec3& max,
    const std::vector<InterNode>& nodes, size_t index,
    int level, int para)
{
    if (para < 4)
        return ConvertInterNodeToQBVHDeep(out_faces, out_nodes, min, max, nodes, index, level);
    else
        return ConvertInterNodeToQBVHParallel(out_faces, out_nodes, min, max, nodes, index, level, para);
}

static inline void CalcQBVHDeep(
    size_t& fsz, size_t& nsz,
    const std::vector<InterNode>& nodes, size_t index,
    int level)
{
    if (index == EMPTY_MASK) return;
    if (nodes.size() <= index) return;
    const InterNode& node = nodes[index];
    size_t sz = GetFaceSize(nodes, index);
    if (sz == 0) return;
    assert(sz != 0);
    if (level != 0 && sz <= MIN_FACE || level >= DIV_BIT * 3)
    { //
        fsz += sz + 1;
    }
    else
    {
        size_t indices[4] = { EMPTY_MASK, EMPTY_MASK, EMPTY_MASK, EMPTY_MASK };

        if (node.nodes[0] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[0]];
            if (cnode.pFace)
            {
                indices[0] = node.nodes[0];
                indices[1] = EMPTY_MASK;
            }
            else
            {
                indices[0] = cnode.nodes[0];
                indices[1] = cnode.nodes[1];
            }
        }
        if (node.nodes[1] != EMPTY_MASK)
        {
            const InterNode& cnode = nodes[node.nodes[1]];
            if (cnode.pFace)
            {
                indices[2] = node.nodes[1];
                indices[3] = EMPTY_MASK;
            }
            else
            {
                indices[2] = cnode.nodes[0];
                indices[3] = cnode.nodes[1];
            }
        }

        nsz += 1;

        size_t cfsz[4] = {};
        size_t cnsz[4] = {};

        CalcQBVHDeep(cfsz[0], cnsz[0], nodes, indices[0], level + 2);
        CalcQBVHDeep(cfsz[1], cnsz[1], nodes, indices[1], level + 2);
        CalcQBVHDeep(cfsz[2], cnsz[2], nodes, indices[2], level + 2);
        CalcQBVHDeep(cfsz[3], cnsz[3], nodes, indices[3], level + 2);

        fsz += cfsz[0] + cfsz[1] + cfsz[2] + cfsz[3];
        nsz += cnsz[0] + cnsz[1] + cnsz[2] + cnsz[3];
    }
}

static inline void CalcQBVHParallel(
    size_t& fsz, size_t& nsz,
    const std::vector<InterNode>& nodes, size_t index,
    int level, int para)
{
    if (index == EMPTY_MASK) return;
    if (nodes.size() <= index) return;
    const InterNode& node = nodes[index];
    size_t sz = GetFaceSize(nodes, index);
    if (sz == 0) return;
    assert(sz != 0);
    if (level != 0 && sz <= MIN_FACE || level >= DIV_BIT * 3)
    { //
        fsz += sz + 1;
    }
    else
    {
        std::vector<size_t> indices(para);

        for (size_t i = 0; i < para; i++)
        {
            indices[i] = EMPTY_MASK;
        }

        ExpandIndices(&indices[0], para, nodes, index);

        nsz += GetSubNodeSize4(para);

        {
            std::vector<size_t> sub_faces(para);
            std::vector<size_t> sub_nodes(para);
            memset(&sub_faces[0], 0, sizeof(size_t) * para);
            memset(&sub_nodes[0], 0, sizeof(size_t) * para);
            {
#pragma omp parallel for
                for (int i = 0; i < para; i++)
                {
                    CalcQBVHDeep(sub_faces[i], sub_nodes[i], nodes, indices[i], level + 2);
                }
            }
            for (int i = 0; i < para; i++)
            {
                fsz += sub_faces[i];
                nsz += sub_nodes[i];
            }
        }
    }
}

static inline void CalcQBVH(
    size_t& fsz, size_t& nsz,
    const std::vector<InterNode>& nodes, size_t index,
    int level, int para)
{
    if (para < 4)
        return CalcQBVHDeep(fsz, nsz, nodes, index, level);
    else
        return CalcQBVHParallel(fsz, nsz, nodes, index, level, para);
}

//------------------------------------------------------------------------------------------------------------------

static inline bool intersect_triangle(isect_t* isect, PCFACE tri, const ray_t& r, float tmin, float tmax)
{
    static const float EPS = 1e-12f;
    float u, v, t;

    const glm::vec3& org = r.org;
    const glm::vec3& dir = r.dir;
    const glm::vec3& p0 = tri->p0;
    const glm::vec3& p1 = tri->p1;
    const glm::vec3& p2 = tri->p2;

    //-e1 = p0-p1
    //glm::vec3 e1 = (p0 - p1); //vA
    const glm::vec3& e1 = tri->e1;

    //-e2 = p0-p2
    //glm::vec3 e2 = (p0 - p2); //vB
    const glm::vec3& e2 = tri->e2;

    //dir = GHI

    glm::vec3 bDir = cross(e2, dir);

    float iM = dot(e1, bDir);

    if (EPS < iM)
    {
        //p0-org
        glm::vec3 vOrg = (p0 - org);

        u = dot(vOrg, bDir);
        if (u < 0 || iM < u) return false;

        glm::vec3 vE = cross(e1, vOrg);

        v = dot(dir, vE);
        if (v < 0 || iM < u + v) return false;

        t = -dot(e2, vE);
        if (t <= 0) return false;
    }
    else if (iM < -EPS)
    {
        //p0-org
        glm::vec3 vOrg = (p0 - org); //JKL

        u = dot(vOrg, bDir);
        if (u > 0 || iM > u) return false;

        glm::vec3 vE = cross(e1, vOrg);

        v = dot(dir, vE);
        if (v > 0 || iM > u + v) return false;

        t = -dot(e2, vE);
        if (t >= 0) return false;
    }
    else
    {
        return false;
    }

    iM = float(1.0) / iM;

    t *= iM;
    if (t <= tmin || tmax <= t) return false;
    u *= iM;
    v *= iM;

    isect->t = t;
    isect->u = u;
    isect->v = v;
    isect->faceid = tri->face_id;
    //memcpy(isect->isect, glm::value_ptr(p), sizeof(float) * 3);
    //memcpy(isect->ns, glm::value_ptr(n), sizeof(float) * 3);
    
    return true;
}



static inline bool intersect_bbox(float tx[2], const float boxes[2][3], const ray_t& r, float tmin, float tmax)
{
    for (int i = 0; i < 3; i++)
    {
        tmin = std::max<float>(tmin, (boxes[r.sign[i]][i] - r.org[i]) * r.idir[i]);
        tmax = std::min<float>(tmax, (boxes[1 - r.sign[i]][i] - r.org[i]) * r.idir[i]);
    }
    if (tmin <= tmax)
    {
        tx[0] = tmin;
        tx[1] = tmax;
        return true;
    }
    return false;
}

static inline int log2i(int n)
{
    static const double LOG2 = log(2.0);
    return (int)floor(log((double)n) / LOG2);
}
static inline int log4i(int n)
{
    static const double LOG4 = log(4.0);
    return (int)floor(log((double)n) / LOG4);
}
#if defined(_OPENMP) && !defined(NO_OMP)
static inline int get_para_nums()
{
    return omp_get_num_threads(); //GetCPUNumber();//?
}
#else
static inline int get_para_nums()
{
    return 1;
}
#endif
static inline int get_para_nums2(int x = get_para_nums())
{
    return std::max<int>(1, (int)pow(2.0f, log2i(x)));
}
static inline int get_para_nums4(int x = get_para_nums())
{
    return std::max<int>(1, (int)pow(4.0f, log4i(x)));
}

class timer_t
{
public:
    void start() {}
    void end(){}
};

class bvh_t
{
public:
    bvh_t(PCFACE* begin, PCFACE* end, const float min[3], const float max[3])
    {
        memcpy(boxes_[0], min, sizeof(float) * 3);
        memcpy(boxes_[1], max, sizeof(float) * 3);

        int para_nums2 = get_para_nums2();
        int para_nums4 = get_para_nums4();
        
        timer_t t;
        t.start();
        size_t sz = end - begin;
        std::vector<code_face> codes;
        { //
            vec3 tmin, tmax;
            vec3 mmin = tovec(min);
            vec3 mmax = tovec(max);
            get_morton_bound(tmin, tmax, mmin, mmax);
            CreateCodeFace(codes, tmin, tmax, begin, end);
        }
        t.end();
       
        //print_log("a:%d\n",t.msec());
        t.start();
        std::vector<InterNode> nodes;
        {
            nodes.reserve(3 * sz);
            CreateInterNode(nodes, codes.begin(), codes.end(), 0, para_nums2);
        }
        t.end();
        //print_log("b:%d\n",t.msec());

        {
            //float cost = GetCostSAH(nodes);
            //print_log("sah 0:%f\n", cost);
        }

        t.start();
        {
            for (int i = 0; i < Iter_; i++)
            {
                OptimizeInterNode(nodes, 0, para_nums2);
            }
        }
        t.end();
        //print_log("c:%d\n",t.msec());

        {
            //float cost = GetCostSAH(nodes);
            //print_log("sah 1:%f\n", cost);
        }

        t.start();
        { //
            size_t fsz = 0;
            size_t nsz = 0;
            CalcQBVH(fsz, nsz, nodes, 0, 0, para_nums4);
            //print_log("pre: fsz:%d, nsz:%d\n",fsz, nsz);
            faces_.reserve(fsz);
            nodes_.reserve(nsz);
            vec3 cmin, cmax;
            ConvertInterNodeToQBVH(faces_, nodes_, cmin, cmax, nodes, 0, 0, para_nums4);
            fsz = faces_.size();
            nsz = nodes_.size();
            //print_log("post: fsz:%d, nsz:%d\n",fsz, nsz);
        }
        t.end();
        //print_log("d:%d\n",t.msec());
    }

    bool intersect_faces(size_t nFirst, isect_t* isect, const ray_t& r, float tnear, float tfar)const
    {
        const PCFACE* faces = &faces_[0];
        bool bRet = false;
        size_t i = nFirst;
        while (faces[i])
        {
            if (intersect_triangle(isect, faces[i], r, tnear, tfar))
            {
                tfar = isect->t;
                bRet = true;
            }
            i++;
        }
        return bRet;
    }

    bool intersect_inner(isect_t* isect, const ray_t& r, float tnear, float tfar)const
    {
        if (nodes_.empty()) return false;

        __m128 sseOrg[3];
        __m128 sseiDir[3];
        int sign[3];
        __m128 sseTMin;
        __m128 sseTMax;

        sseOrg[0] = _mm_set1_ps((r.org[0]));
        sseOrg[1] = _mm_set1_ps((r.org[1]));
        sseOrg[2] = _mm_set1_ps((r.org[2]));

        sseiDir[0] = _mm_set1_ps((r.idir[0]));
        sseiDir[1] = _mm_set1_ps((r.idir[1]));
        sseiDir[2] = _mm_set1_ps((r.idir[2]));

        sign[0] = r.sign[0];
        sign[1] = r.sign[1];
        sign[2] = r.sign[2];

        sseTMin = _mm_set1_ps((tnear));
        sseTMax = _mm_set1_ps((tfar));

        const SIMDBVHNode* nodes = &nodes_[0];

        int todoNode = 0;
        size_t nodeStack[STACK_SIZE];
        nodeStack[0] = 0;

        bool bRet = false;
        while (todoNode >= 0)
        {
            size_t idx = nodeStack[todoNode];

            if (IsBranch(idx))
            {
                todoNode--; //pop stack
                const SIMDBVHNode& node = nodes[idx];

                int HitMask = test_AABB(node.bboxes, sseOrg, sseiDir, sign, sseTMin, sseTMax);

                if (HitMask)
                {
                    int NodeIdx = (sign[node.axis_top] << 2) | (sign[node.axis_left] << 1) | (sign[node.axis_right]);
                    int Order = OrderTable[HitMask * 8 + NodeIdx];

                    while (!(Order & 0x4))
                    {
                        todoNode++;
                        nodeStack[todoNode] = node.children[Order & 0x3];
                        Order >>= 4;
                    }
                }
            }
            else
            { // if(IsLeaf(idx))
                todoNode--;
                if (IsEmpty(idx)) continue;

                size_t fi = GetFaceFirst(idx); //convert face index.

                if (this->intersect_faces(fi, isect, r, tnear, tfar))
                {
                    bRet = true;
                    tfar = isect->t;
                    sseTMax = _mm_set1_ps((tfar));
                }
            }

            assert(todoNode < STACK_SIZE);
        }

        return bRet;
    }

    bool intersect(isect_t* isect, const ray_t& r, float tnear, float tfar) const
    {
        float tx[2] = {};
        if (intersect_bbox(tx, boxes_, r, tnear, tfar))
        {
            bool bRet = false;
            tnear = tx[0];
            tfar = tx[1];
            return intersect_inner(isect, r, tnear, tfar);
        }
        return false;
    }
protected:
    float boxes_[2][3];
    std::vector<PCFACE> faces_;
    aligned_vector<SIMDBVHNode> nodes_;
};


static std::shared_ptr<mesh_t> g_mesh;
static std::shared_ptr<bvh_t> g_bvh;

static void get_minmax(float min[3], float max[3], PCFACE* a, PCFACE* b)
{
    static const float far_ = std::numeric_limits<float>::max();
    min[0] = +far_;
    min[1] = +far_;
    min[2] = +far_;
    max[0] = -far_;
    max[1] = -far_;
    max[2] = -far_;
    for (PCFACE* it = a; it != b; it++)
    {
        PCFACE tri = *it;
        for (int j = 0; j < 3; j++)
        {
            float x0 = tri->p0[j];
            float x1 = tri->p1[j];
            float x2 = tri->p2[j];
            min[j] = std::min<float>(min[j], x0);
            min[j] = std::min<float>(min[j], x1);
            min[j] = std::min<float>(min[j], x2);

            max[j] = std::max<float>(max[j], x0);
            max[j] = std::max<float>(max[j], x1);
            max[j] = std::max<float>(max[j], x2);
        }
    }
}

void preprocess(
    const float* vertices,
    size_t numVerts,
    const float* normals,
    size_t numNormals,
    const uint32_t* indices,
    size_t numFace)
{
    std::shared_ptr<mesh_t> mesh(new mesh_t());
    mesh->vertices.resize(numVerts * 3);
    memcpy(&mesh->vertices[0], vertices, sizeof(float) * numVerts * 3);
    mesh->normals.resize(numNormals * 3);
    memcpy(&mesh->normals[0], normals, sizeof(float) * numNormals * 3);
    mesh->triangles.resize(numFace);
    for (size_t i = 0; i < numFace; i++)
    {
        uint32_t v0 = indices[6 * i + 0];
        uint32_t v1 = indices[6 * i + 2];
        uint32_t v2 = indices[6 * i + 4];
        uint32_t n0 = indices[6 * i + 1];
        uint32_t n1 = indices[6 * i + 3];
        uint32_t n2 = indices[6 * i + 5];
        mesh->triangles[i].p0 = tovec(&mesh->vertices[uint32_t(3 * v0)]);
        mesh->triangles[i].p1 = tovec(&mesh->vertices[uint32_t(3 * v1)]);
        mesh->triangles[i].p2 = tovec(&mesh->vertices[uint32_t(3 * v2)]);
        mesh->triangles[i].n0 = tovec(&mesh->normals[uint32_t(3 * n0)]);
        mesh->triangles[i].n1 = tovec(&mesh->normals[uint32_t(3 * n1)]);
        mesh->triangles[i].n2 = tovec(&mesh->normals[uint32_t(3 * n2)]);
        mesh->triangles[i].e1 = mesh->triangles[i].p0 - mesh->triangles[i].p1;
        mesh->triangles[i].e2 = mesh->triangles[i].p0 - mesh->triangles[i].p2;
        mesh->triangles[i].center = (mesh->triangles[i].p0 + mesh->triangles[i].p1 + mesh->triangles[i].p2) * (1.0f / 3);
        mesh->triangles[i].face_id = (uint32_t)i;
    }
    g_mesh = mesh;
    
    std::vector<const triangle_t*> p_triangles(numFace);
    for (size_t i = 0; i < numFace; i++)
    {
        p_triangles[i] = &mesh->triangles[i];
    }   

    float min[3] = {};
    float max[3] = {};
    get_minmax(min, max, &p_triangles[0], (&p_triangles[0]) + numFace);
    std::shared_ptr<bvh_t> bvh(new bvh_t(&p_triangles[0], (&p_triangles[0]) + numFace, min, max));
    g_bvh = bvh;
}


inline static float safe_invert(float x)
{
   
    if (std::abs(x) < 1e-8f)
    {
        if (x < 0)
            return -std::numeric_limits<float>::max();
        else
            return +std::numeric_limits<float>::max();
    }
    else
    {
        return float(1) / x;
    }
}

void intersect(
    Ray* rays,
    size_t numRay,
    bool hitany)
{
    if (g_bvh.get())
    {
        {
            std::vector<ray_t> rbs(numRay);
            //#pragma omp parallel for
            for (int32_t nr = 0; nr < numRay; ++nr)
            {
                Ray& ra = rays[nr];
                ray_t& rb = rbs[nr];
                rb.org = tovec(ra.pos);
                rb.dir = tovec(ra.dir);
                rb.idir[0] = safe_invert(ra.dir[0]);
                rb.idir[1] = safe_invert(ra.dir[1]);
                rb.idir[2] = safe_invert(ra.dir[2]);
                rb.sign[0] = ra.dir[0] < 0 ? 1 : 0;
                rb.sign[1] = ra.dir[1] < 0 ? 1 : 0;
                rb.sign[2] = ra.dir[2] < 0 ? 1 : 0;
            }

            //#pragma omp parallel for
            for (int32_t nr = 0; nr < numRay; ++nr)
            {
                Ray& ra = rays[nr];
                ray_t& rb = rbs[nr];
                isect_t isect = {};
                bool bHit = g_bvh->intersect(&isect, rb, ra.tnear, ra.tfar);
                if (bHit)
                {
                    ra.isisect = true;
                    ra.faceid = isect.faceid;

                    glm::vec3 p = rb.org + isect.t * rb.dir;

                    const triangle_t& tri = g_mesh->triangles[isect.faceid];
                    const glm::vec3& n0 = tri.n0;
                    const glm::vec3& n1 = tri.n1;
                    const glm::vec3& n2 = tri.n2;


                    float u = isect.u;
                    float v = isect.v;
                    float w = 1.0f - (u + v);
                    glm::vec3 n = normalize((w)* n0 + (u)* n1 + (v)* n2);
                    //glm::vec3 n = normalize(cross(-e1, -e2));

                    if (dot(rb.dir, n) > 0)
                    {
                        n = -n;
                    }
                    memcpy(ra.isect, glm::value_ptr(p), sizeof(float) * 3);
                    memcpy(ra.ns, glm::value_ptr(n), sizeof(float) * 3);
                }
                else
                {
                    ra.isisect = false;
                }
                hitany |= bHit;
            }
        }
    }
}