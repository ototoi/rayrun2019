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

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
    glm::vec3 org;
    glm::vec3 dir;
    glm::vec3 idir;
    int sign[3];
} ray_t;

static inline glm::vec3 tovec(const float f[3])
{
    return glm::vec3(f[0], f[1], f[2]);
}

static inline bool intersect_triangle(isect_t* isect, const triangle_t* tri, const ray_t& r, float tmin, float tmax)
{
    static const float EPSILON = 1e-12f;
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

    if (EPSILON < iM)
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
    else if (iM < -EPSILON)
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

class bvh_base
{
public:
    virtual ~bvh_base() {}
    virtual bool intersect(isect_t* isect, const ray_t& r, float tnear, float tfar)const = 0;

};

class bvh_leaf : public bvh_base
{
public:
    bvh_leaf(const std::vector<const triangle_t*>& triangles, const float min[3], const float max[3])
        :triangles_(triangles)
    {
        memcpy(boxes_[0], min, sizeof(float) * 3);
        memcpy(boxes_[1], max, sizeof(float) * 3);
    }
    bool intersect(isect_t* isect, const ray_t& r, float tnear, float tfar)const
    {
        float tx[2] = {};
        if (intersect_bbox(tx, boxes_, r, tnear, tfar))
        {
            bool bRet = false;
            const PCFACE* triangles = &triangles_[0];
            size_t sz = triangles_.size();
            for (size_t i = 0; i < sz; i++)
            {
                if (intersect_triangle(isect, triangles[i], r, tnear, tfar))
                {
                    tfar = isect->t;
                    bRet |= true;
                }
            }
            return bRet;
        }
        return false;
    }
protected:
    std::vector<const triangle_t*> triangles_;
    float boxes_[2][3];
};

class bvh_branch : public bvh_base
{
public:
    bvh_branch(std::shared_ptr<bvh_base> children[2], int plane, const float min[3], const float max[3])
    {
        children_[0] = children[0];
        children_[1] = children[1];
        plane_ = plane;
        memcpy(boxes_[0], min, sizeof(float) * 3);
        memcpy(boxes_[1], max, sizeof(float) * 3);
    }

    bool intersect(isect_t* isect, const ray_t& r, float tnear, float tfar)const
    {
        float tx[2] = {};
        if (intersect_bbox(tx, boxes_, r, tnear, tfar))
        {
            bool bRet = false;
            tnear = tx[0];
            tfar = tx[1];
            int nFirst = r.sign[this->plane_];
            int nSecond = 1 - nFirst;

            //if (children_[nFirst].get())
            {
                if (children_[nFirst]->intersect(isect, r, tnear, tfar))
                {
                    tfar = isect->t;
                    bRet |= true;
                }
            }
            //if (children_[nSecond].get())
            {
                if (children_[nSecond]->intersect(isect, r, tnear, tfar))
                {
                    tfar = isect->t;
                    bRet |= true;
                }
            }
            return bRet;
        }
        return false;
    }
private:
    float boxes_[2][3];
    int plane_;
    std::shared_ptr<bvh_base> children_[2];
};


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

struct FaceSorter
{
    FaceSorter(int plane)
        :plane_(plane)
    {}
    bool operator()(PCFACE a, PCFACE b)
    {
        float aa = a->p0[plane_] + a->p1[plane_] + a->p2[plane_];
        float bb = b->p0[plane_] + b->p1[plane_] + b->p2[plane_];
        return aa < bb;
    }
    int plane_;
};

static std::shared_ptr<bvh_base> construct_bvh(PCFACE* a, PCFACE* b)
{
    size_t sz = b - a;
    if (sz <= 10)
    {
        float min[3] = {};
        float max[3] = {};
        get_minmax(min, max, a, b);
        std::vector<const triangle_t*> tmp; tmp.reserve(sz);
        for (const PCFACE* it = a; it != b; it++)
        {
            tmp.push_back(*it);
        }
        std::shared_ptr<bvh_base> bvh(new bvh_leaf(tmp, min, max));
        return bvh;
    }
    else
    {
        float min[3] = {};
        float max[3] = {};
        get_minmax(min, max, a, b);
        float wid[3] = {max[0]-min[0], max[1] - min[1], max[2] - min[2]};
        int plane = 0;
        if (wid[1] > wid[plane])plane = 1;
        if (wid[2] > wid[plane])plane = 2;
        std::sort(a, b, FaceSorter(plane));
        PCFACE* c = a + sz / 2;

        std::shared_ptr<bvh_base> children[2];
        children[0] = construct_bvh(a, c);
        children[1] = construct_bvh(c, b);
        std::shared_ptr<bvh_base> bvh(new bvh_branch(children, plane, min, max));
        return bvh;
    }
    
}

static std::shared_ptr<mesh_t> g_mesh;
static std::shared_ptr<bvh_base> g_bvh;


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
        mesh->triangles[i].face_id = (uint32_t)i;
    }
    g_mesh = mesh;

    std::vector<const triangle_t*> p_triangles(numFace);
    for (size_t i = 0; i < numFace; i++)
    {
        p_triangles[i] = &mesh->triangles[i];
    }   
    g_bvh = construct_bvh(&p_triangles[0], (&p_triangles[0]) + numFace);
}

inline static int get_phase(const float dir[3])
{
    int phase = 0;
    if (dir[0] < 0) phase |= 1;
    if (dir[1] < 0) phase |= 2;
    if (dir[2] < 0) phase |= 4;
    return phase;
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
        isect_t isect;
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