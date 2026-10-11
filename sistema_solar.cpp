// =============================================================================
//  SIMULADOR DEL SISTEMA SOLAR  -  C++ + Raylib (un solo archivo, sin texturas externas)
// =============================================================================
//  COMPILAR (w64devkit / MinGW en Windows; ajusta las rutas a tu raylib):
//      g++ sistema_solar.cpp -o sistema_solar.exe -std=c++17 -O2 -I C:/raylib/raylib/src -L C:/raylib/raylib/src -lraylib -lopengl32 -lgdi32 -lwinmm
//  Linux:  g++ sistema_solar.cpp -o sistema_solar -std=c++17 -O2 -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
//  Requisitos: raylib 5.0 o superior, OpenGL 3.3, y guardar este archivo en UTF-8 (lleva tildes).
//
//  FISICA (unidades UA / dia / masa solar)
//   - N cuerpos: Sol, planetas, enanos y asteroides grandes con integrador leapfrog (orden 2, simplectico, reversible).
//   - Cada planeta lleva sus lunas (26 en total) en coordenadas planetocentricas con integrador de 4o orden (Yoshida):
//     achatamiento J2 del planeta, atraccion mutua entre lunas y marea del Sol (incluye la resonancia Io-Europa-Ganimedes).
//   - ~10 000 particulas de prueba: cinturon principal con huecos de Kirkwood, hildas, troyanos de Jupiter, Kuiper, centauros y NEOs.
//   - Cometas (Halley, Encke, 67P, Hale-Bopp) sin masa, con subpasos adaptativos cerca del Sol.
//   - Condiciones iniciales a la fecha real: elementos medios de Standish (JPL) + teoria lunar de Meeus (fase lunar real).
//   - Todos los cuerpos rotan con su periodo e inclinacion de eje (Venus, Urano y Pluton retrogrados; lunas sincronicas).
//   - Un regulador de rendimiento baja la precision de las lunas (no la fluidez) cuando se acelera mucho el tiempo.
//
//  GRAFICOS: shaders GLSL 330 procedurales: planetas con relieve, atmosferas por dispersion, anillos de Saturno con sombras,
//  eclipses, cielo con la Via Lactea en su orientacion galactica real, Sol granulado y colas de cometas.
//  CONTROLES: pulsa H dentro del programa (raton, tactil y teclado).
// =============================================================================
#include "raylib.h"
#include "rlgl.h"
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cstdarg>

// ======================= [FISICA:INICIO] =======================
struct V3 { double x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 operator*(double s, V3 a) { return {a.x * s, a.y * s, a.z * s}; }
inline V3& operator+=(V3& a, V3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline V3& operator-=(V3& a, V3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double vlen(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 vnorm(V3 a) { double l = vlen(a); return l > 1e-300 ? a * (1.0 / l) : V3{0, 0, 1}; }
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

const double kPI = 3.14159265358979323846;
const double kDEG = kPI / 180.0;
const double AU_KM = 149597870.700;
inline double gmConv(double km3s2) { return km3s2 * 86400.0 * 86400.0 / (AU_KM * AU_KM * AU_KM); } // km^3/s^2 -> UA^3/dia^2

// Matriz 3x3 por columnas (imagen de los ejes locales)
struct Mat3 { V3 c0{1, 0, 0}, c1{0, 1, 0}, c2{0, 0, 1}; };
inline V3 mulM(const Mat3& m, V3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
inline V3 mulMT(const Mat3& m, V3 v) { return {dot(m.c0, v), dot(m.c1, v), dot(m.c2, v)}; }

// Marco ecuatorial de un cuerpo: z = eje de giro, x = nodo del ecuador con la ecliptica
Mat3 eqFrame(V3 spin) {
    V3 z = vnorm(spin);
    V3 x = cross(V3{0, 0, 1}, z);
    if (vlen(x) < 1e-9) x = V3{1, 0, 0};
    x = vnorm(x);
    V3 y = cross(z, x);
    return {x, y, z};
}
// Polo (AR, Dec ICRF) -> vector en la ecliptica J2000
V3 poleFromRaDec(double ra, double dec) {
    double a = ra * kDEG, d = dec * kDEG, eps = 23.4392911 * kDEG;
    double xe = std::cos(d) * std::cos(a), ye = std::cos(d) * std::sin(a), ze = std::sin(d);
    return {xe, ye * std::cos(eps) + ze * std::sin(eps), -ye * std::sin(eps) + ze * std::cos(eps)};
}

// ---------------- Mecanica kepleriana ----------------
double solveKepler(double M, double e) {
    M = std::fmod(M, 2 * kPI);
    if (M < -kPI) M += 2 * kPI;
    if (M > kPI) M -= 2 * kPI;
    double E = (e < 0.8) ? M : (M >= 0 ? kPI : -kPI);
    for (int k = 0; k < 60; k++) {
        double d = (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
        E -= d;
        if (std::fabs(d) < 1e-14) break;
    }
    return E;
}
Mat3 rotPQW(double Om, double inc, double w) {
    double cO = std::cos(Om), sO = std::sin(Om), ci = std::cos(inc), si = std::sin(inc), cw = std::cos(w), sw = std::sin(w);
    V3 P = {cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si};
    V3 Q = {-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si};
    V3 W = {sO * si, -cO * si, ci};
    return {P, Q, W};
}
void stateFromElements(double mu, double a, double e, double inc, double Om, double w, double M, V3& r, V3& v) {
    double E = solveKepler(M, e), cE = std::cos(E), sE = std::sin(E), sq = std::sqrt(1 - e * e);
    double x = a * (cE - e), y = a * sq * sE;
    double n = std::sqrt(mu / (a * a * a)), f = n * a / (1 - e * cE);
    Mat3 R = rotPQW(Om, inc, w);
    r = R.c0 * x + R.c1 * y;
    v = R.c0 * (-f * sE) + R.c1 * (f * sq * cE);
}
struct Elems { double a = 0, e = 0, inc = 0, Om = 0, w = 0, M = 0, nu = 0, T = 0; bool bound = false; V3 P, Q, h; };
Elems elementsFromState(V3 r, V3 v, double mu) {
    Elems el;
    double rn = vlen(r);
    V3 h = cross(r, v);
    double hn = vlen(h);
    if (hn < 1e-300 || rn < 1e-300) return el;
    V3 ev = cross(v, h) * (1.0 / mu) - r * (1.0 / rn);
    double e = vlen(ev), en = 0.5 * dot(v, v) - mu / rn;
    el.e = e; el.h = h;
    V3 W = h * (1.0 / hn);
    el.inc = std::acos(clampd(W.z, -1, 1));
    V3 nv = {-h.y, h.x, 0};
    double nn = vlen(nv);
    V3 nh = nn > 1e-12 * hn ? nv * (1.0 / nn) : V3{1, 0, 0};
    el.Om = nn > 1e-12 * hn ? std::atan2(nv.y, nv.x) : 0.0;
    V3 Pd = e > 1e-10 ? ev * (1.0 / e) : (nn > 1e-12 * hn ? nh : vnorm(r));
    V3 Qd = cross(W, Pd);
    el.P = Pd; el.Q = Qd;
    el.w = std::atan2(dot(Pd, cross(W, nh)), dot(Pd, nh));
    if (el.w < 0) el.w += 2 * kPI;
    if (el.Om < 0) el.Om += 2 * kPI;
    el.nu = std::atan2(dot(r, Qd), dot(r, Pd));
    el.bound = (en < 0 && e < 0.9999);
    if (el.bound) {
        el.a = -mu / (2 * en);
        double E = 2 * std::atan2(std::sqrt(1 - e) * std::sin(el.nu / 2), std::sqrt(1 + e) * std::cos(el.nu / 2));
        el.M = E - e * std::sin(E);
        el.T = 2 * kPI * std::sqrt(el.a * el.a * el.a / mu);
    }
    return el;
}

// ---------------- Estructuras ----------------
// Aspecto visual procedural de un cuerpo (lo interpreta el shader)
struct Look {
    int type = 0;                         // 0 rocoso, 1 Tierra, 3 gigante gaseoso, 4 nubes/neblina, 5 Sol
    float c1[3] = {0.5f, 0.5f, 0.5f}, c2[3] = {0.35f, 0.35f, 0.35f}, c3[3] = {0.75f, 0.75f, 0.75f};
    float par[4] = {1.f, 0.5f, 0.f, 0.f};  // escala de detalle, crateres/contraste, hielo/tormenta, extra
    float atm[5] = {0, 0, 0, 0, 0};       // rgb dispersion, grosor relativo, densidad optica
    float ringIn = 0, ringOut = 0;        // radios planetarios; 0 = sin anillos
    int ringType = 0;
    float lump = 0;                       // >0: cuerpo irregular (normales facetadas)
};
enum BodyKind { K_STAR, K_PLANET, K_MOON, K_SMALL, K_COMET };
struct Body {
    std::string name, key;
    BodyKind kind = K_PLANET;
    int parent = -1, prim = -1, moonSlot = -1;
    bool alive = true, massless = false, userCreated = false, dwarf = false;
    double gm = 0, gm0 = 0;               // UA^3/dia^2
    double radiusKm = 1, radius0 = 1, flat = 0, flat0 = 0, j2 = 0;
    V3 axes{1, 1, 1};                     // semiejes relativos (cuerpos irregulares)
    V3 pos, vel;                          // absolutos (baricentricos, ecliptica J2000)
    V3 spin{0, 0, 1};
    double spinRate = 0, spinPhase = 0;   // rad/dia, rad
    double spinRate0 = 0;
    V3 spin0{0, 0, 1};
    bool tidalLock = false, refEcliptic = true, rings = false;
    double a0 = 0, e0 = 0, i0 = 0;        // elementos iniciales (para restaurar)
    Color col{255, 255, 255, 255};        // color de orbita / marcador
    Look look;
    double dispPhase = 0;                 // fase de giro mostrada (limitada para evitar estroboscopio)
};
struct Prim {   // punto masivo que se mueve en el campo del Sol (Sol, baricentro de un sistema, cuerpo suelto)
    V3 x, v, a;
    double gm = 0;
    int body = -1;
    bool alive = true, massless = false;
    std::vector<int> moons;
    std::vector<V3> r, w, acc;            // estado planetocentrico de las lunas
    double hMax = 1e9;
};
struct Particle { V3 x, v, a; float size; unsigned char type, shade; bool alive; };

struct Sim {
    std::vector<Body> b;
    std::vector<Prim> p;
    std::vector<Particle> sw;
    double t = 0, jd0 = 0, swarmAccum = 0;
    bool accValid = false, swarmValid = false;
    std::vector<V3> x0, v0, psx;
    std::vector<double> inv3;
    double Hmax = 0.5, moonDiv = 60.0;
    bool useJ2 = true;
    std::function<void()> stepHook;
    std::vector<double> psg;

    static V3 hermite(V3 xa, V3 va, V3 xb, V3 vb, double H, double s) {
        double s2 = s * s, s3 = s2 * s;
        return xa * (2 * s3 - 3 * s2 + 1) + va * (H * (s3 - 2 * s2 + s)) + xb * (-2 * s3 + 3 * s2) + vb * (H * (s3 - s2));
    }
    Mat3 frameOf(int i) const { return b[i].refEcliptic ? Mat3{} : eqFrame(b[b[i].parent].spin); }

    void computePrimAcc() {
        size_t n = p.size();
        for (auto& q : p) q.a = V3{};
        for (size_t i = 0; i < n; i++) {
            if (!p[i].alive || p[i].massless) continue;
            for (size_t j = i + 1; j < n; j++) {
                if (!p[j].alive || p[j].massless) continue;
                V3 d = p[j].x - p[i].x;
                double r2 = dot(d, d) + 1e-14, f = 1.0 / (r2 * std::sqrt(r2));
                p[j].a -= d * (p[i].gm * f);
                p[i].a += d * (p[j].gm * f);
            }
        }
    }
    void moonAcc(Prim& pr, const V3& D, const Mat3& F, std::vector<V3>& acc) {
        int nm = (int)pr.moons.size();
        const Body& pl = b[pr.body];
        double gmp = pl.gm, gms = b[0].gm, Req = pl.radiusKm / AU_KM;
        inv3.resize(nm);
        V3 S;
        for (int j = 0; j < nm; j++) {
            const Body& m = b[pr.moons[j]];
            if (!m.alive) continue;
            double r2 = dot(pr.r[j], pr.r[j]) + 1e-30, i3 = 1.0 / (r2 * std::sqrt(r2));
            inv3[j] = i3;
            S += pr.r[j] * (m.gm * i3);
        }
        double Dn = dot(D, D), iD3 = 1.0 / (Dn * std::sqrt(Dn));
        V3 Dterm = D * (gms * iD3);
        for (int i = 0; i < nm; i++) {
            if (!b[pr.moons[i]].alive) { acc[i] = V3{}; continue; }
            V3 a = pr.r[i] * (-gmp * inv3[i]) - S;
            for (int j = 0; j < nm; j++) {
                if (j == i || !b[pr.moons[j]].alive) continue;
                V3 d = pr.r[j] - pr.r[i];
                double d2 = dot(d, d) + 1e-30;
                a += d * (b[pr.moons[j]].gm / (d2 * std::sqrt(d2)));
            }
            V3 dd = D - pr.r[i];
            double q = dot(dd, dd);
            a += dd * (gms / (q * std::sqrt(q))) - Dterm;
            if (useJ2 && pl.j2 != 0) {      // achatamiento ecuatorial del planeta (J2)
                V3 u = mulMT(F, pr.r[i]);
                double r2 = dot(u, u), r5 = r2 * r2 * std::sqrt(r2), z2 = u.z * u.z / r2;
                double k = -1.5 * pl.j2 * gmp * Req * Req / r5;
                a += mulM(F, V3{k * u.x * (1 - 5 * z2), k * u.y * (1 - 5 * z2), k * u.z * (3 - 5 * z2)});
            }
            acc[i] = a;
        }
    }
    void advanceMoons(int k, double H) {
        Prim& pr = p[k];
        int nm = (int)pr.moons.size();
        if (!nm) return;
        int m = (int)std::ceil(std::fabs(H) / pr.hMax);
        if (m < 1) m = 1;
        double hs = H / m;
        Prim& sun = p[0];
        Mat3 F = eqFrame(b[pr.body].spin);
        auto Dat = [&](double s) -> V3 {
            V3 B = hermite(x0[k], v0[k], pr.x, pr.v, H, s);
            V3 Sx = hermite(x0[0], v0[0], sun.x, sun.v, H, s);
            V3 off;
            for (int i = 0; i < nm; i++) if (b[pr.moons[i]].alive) off += pr.r[i] * (b[pr.moons[i]].gm / pr.gm);
            return Sx - (B - off);
        };
        pr.acc.resize(nm);
        moonAcc(pr, Dat(0), F, pr.acc);
        // integrador simplectico de 4o orden (Yoshida): tres pasos "leapfrog" con coeficientes w1, w0, w1
        static const double Wc[3] = {1.3512071919596578, -1.7024143839193153, 1.3512071919596578};
        double ts = 0;
        for (int s = 0; s < m; s++) {
            for (int k = 0; k < 3; k++) {
                double h = Wc[k] * hs;
                for (int i = 0; i < nm; i++) { pr.w[i] += pr.acc[i] * (h * 0.5); pr.r[i] += pr.w[i] * h; }
                ts += h;
                moonAcc(pr, Dat(ts / H), F, pr.acc);
                for (int i = 0; i < nm; i++) pr.w[i] += pr.acc[i] * (h * 0.5);
            }
        }
    }
    // Cuerpos sin masa (cometas, asteroides con nombre): subpasos adaptativos cerca del Sol
    void advanceTest(int k, double H) {
        Prim& q = p[k];
        auto accAt = [&](const V3& x, double s) -> V3 {
            V3 a;
            for (size_t j = 0; j < p.size(); j++) {
                const Prim& m = p[j];
                if (!m.alive || m.massless || m.gm <= 0) continue;
                V3 d = hermite(x0[j], v0[j], m.x, m.v, H, s) - x;
                double r2 = dot(d, d) + 1e-14;
                a += d * (m.gm / (r2 * std::sqrt(r2)));
            }
            return a;
        };
        double rs = std::max(vlen(q.x - p[0].x), 1e-3), tdyn = std::sqrt(rs * rs * rs / std::max(p[0].gm, 1e-12));
        int m = (int)clampd(std::ceil(std::fabs(H) / (0.03 * tdyn)), 1, 300);
        double hs = H / m;
        V3 a = accAt(q.x, 0);
        for (int s = 0; s < m; s++) {
            q.v += a * (hs * 0.5); q.x += q.v * hs;
            a = accAt(q.x, (s + 1.0) / m);
            q.v += a * (hs * 0.5);
        }
    }
    void stepCoarse(double H) {
        size_t n = p.size();
        if (!accValid) { computePrimAcc(); accValid = true; }
        x0.resize(n); v0.resize(n);
        for (size_t k = 0; k < n; k++) { x0[k] = p[k].x; v0[k] = p[k].v; }
        for (auto& q : p) if (q.alive && !q.massless) q.v += q.a * (H * 0.5);
        for (auto& q : p) if (q.alive && !q.massless) q.x += q.v * H;
        computePrimAcc();
        for (auto& q : p) if (q.alive && !q.massless) q.v += q.a * (H * 0.5);
        for (size_t k = 0; k < n; k++) {
            if (!p[k].alive) continue;
            if (p[k].massless) advanceTest((int)k, H);
            else if (!p[k].moons.empty()) advanceMoons((int)k, H);
        }
        t += H;
        swarmAccum += H;
    }
    void gatherMassive() {
        psx.clear(); psg.clear();
        for (auto& q : p) if (q.alive && q.gm > 0) { psx.push_back(q.x); psg.push_back(q.gm); }
    }
    V3 swarmAcc(const V3& x) const {
        V3 a;
        for (size_t k = 0; k < psx.size(); k++) {
            V3 d = psx[k] - x;
            double r2 = dot(d, d) + 1e-8;
            a += d * (psg[k] / (r2 * std::sqrt(r2)));
        }
        return a;
    }
    void swarmStep(double Hs) {
        gatherMassive();
        if (!swarmValid) { for (auto& q : sw) if (q.alive) q.a = swarmAcc(q.x); swarmValid = true; }
        for (auto& q : sw) if (q.alive) { q.v += q.a * (Hs * 0.5); q.x += q.v * Hs; }
        V3 sun = p[0].x;
        for (auto& q : sw) {
            if (!q.alive) continue;
            q.a = swarmAcc(q.x);
            q.v += q.a * (Hs * 0.5);
            double r = vlen(q.x - sun);
            if (r < 0.0047 || r > 5000 || r != r) q.alive = false;
        }
    }
    void syncBodies() {
        for (auto& pr : p) {
            if (!pr.alive) continue;
            Body& pl = b[pr.body];
            V3 off, voff;
            for (size_t k = 0; k < pr.moons.size(); k++) {
                const Body& m = b[pr.moons[k]];
                if (!m.alive) continue;
                double mu = m.gm / pr.gm;
                off += pr.r[k] * mu; voff += pr.w[k] * mu;
            }
            pl.pos = pr.x - off; pl.vel = pr.v - voff;
            for (size_t k = 0; k < pr.moons.size(); k++) {
                Body& m = b[pr.moons[k]];
                m.pos = pl.pos + pr.r[k]; m.vel = pl.vel + pr.w[k];
            }
        }
    }
    void advance(double simDt, double budget, double& done, bool& truncated) {
        truncated = false; done = 0;
        if (simDt == 0) { syncBodies(); return; }
        int n = (int)std::ceil(std::fabs(simDt) / Hmax);
        if (n < 1) n = 1;
        double H = simDt / n, swarmMin = clampd(std::fabs(simDt) / 4.0, 0.5, 10.0);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) {
            stepCoarse(H); done += H;
            if (sw.empty()) swarmAccum = 0;
            else if (std::fabs(swarmAccum) >= swarmMin) { swarmStep(swarmAccum); swarmAccum = 0; }
            if (stepHook) { syncBodies(); stepHook(); }
            if ((i & 3) == 3 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > budget) { truncated = true; break; }
        }
        syncBodies();
    }
    // masa total de cada sistema y paso maximo de las lunas
    void recomputeMasses() {
        for (auto& pr : p) {
            if (!pr.alive) continue;
            Body& pl = b[pr.body];
            double g = pl.massless ? 0.0 : pl.gm, Tmin = 1e9;
            for (size_t k = 0; k < pr.moons.size(); k++) if (b[pr.moons[k]].alive) g += b[pr.moons[k]].gm;
            pr.gm = pl.massless ? 0.0 : g;
            pr.massless = pl.massless;
            for (size_t k = 0; k < pr.moons.size(); k++) {
                const Body& m = b[pr.moons[k]];
                if (!m.alive) continue;
                double mu = pl.gm + m.gm, rn = vlen(pr.r[k]);
                double inv = 2.0 / rn - dot(pr.w[k], pr.w[k]) / mu;
                double a = inv > 1e-12 ? 1.0 / inv : rn;
                Tmin = std::min(Tmin, 2 * kPI * std::sqrt(a * a * a / mu));
            }
            pr.hMax = Tmin / moonDiv;
        }
        accValid = false; swarmValid = false;
    }
    // Estado de un cuerpo respecto a su centro de referencia
    bool relState(int i, V3& r, V3& v, double& mu, V3& center, Mat3& F) {
        Body& bd = b[i];
        if (bd.kind == K_STAR) return false;
        if (bd.kind == K_MOON) {
            Prim& pr = p[b[bd.parent].prim];
            r = pr.r[bd.moonSlot]; v = pr.w[bd.moonSlot];
            mu = b[bd.parent].gm + bd.gm; center = b[bd.parent].pos; F = frameOf(i);
            return true;
        }
        Prim& pr = p[bd.prim];
        r = pr.x - p[0].x; v = pr.v - p[0].v; mu = b[0].gm + pr.gm; center = p[0].x; F = Mat3{};
        return true;
    }
    void setRelState(int i, V3 r, V3 v) {
        Body& bd = b[i];
        if (bd.kind == K_MOON) { Prim& pr = p[b[bd.parent].prim]; pr.r[bd.moonSlot] = r; pr.w[bd.moonSlot] = v; }
        else { Prim& pr = p[bd.prim]; pr.x = p[0].x + r; pr.v = p[0].v + v; }
        accValid = false; swarmValid = false;
        recomputeMasses();
        syncBodies();
    }
    // Cambia a, e, i (respecto a su marco) conservando orientacion y fase
    void applyElements(int i, double a, double e, double inc) {
        V3 r, v, c; double mu; Mat3 F;
        if (!relState(i, r, v, mu, c, F)) return;
        Elems el = elementsFromState(mulMT(F, r), mulMT(F, v), mu);
        if (!el.bound) { el.Om = 0; el.w = 0; el.M = 0; }
        V3 nr, nv;
        stateFromElements(mu, a, clampd(e, 0, 0.97), inc, el.Om, el.w, el.M, nr, nv);
        setRelState(i, mulM(F, nr), mulM(F, nv));
    }
    void circularize(int i) {
        V3 r, v, c; double mu; Mat3 F;
        if (!relState(i, r, v, mu, c, F)) return;
        V3 h = vnorm(cross(r, v));
        V3 vt = vnorm(cross(h, r)) * std::sqrt(mu / vlen(r));
        setRelState(i, r, vt);
    }
    void restoreOrbit(int i) {
        if (b[i].kind == K_STAR || b[i].a0 <= 0) return;
        applyElements(i, b[i].a0, b[i].e0, b[i].i0);
    }
};
// ======================= [FISICA:FIN] =======================

// ============================================================================
//  DATOS DEL SISTEMA SOLAR Y CONSTRUCCION DEL MODELO
//  - Planetas: elementos medios de Standish (JPL, validos 1800-2050) a la fecha.
//  - Luna: posicion inicial de la teoria lunar de Meeus (fase real el dia de hoy).
//  - Lunas, enanos y cometas: elementos orbitales publicados (fase aproximada,
//    salvo Halley/Encke/67P/Hale-Bopp, calculados desde su fecha de perihelio).
// ============================================================================
inline double jdFromDate(int y, int m, int d, double hour = 0) {
    if (m <= 2) { y--; m += 12; }
    int A = y / 100, B = 2 - A + A / 4;
    return std::floor(365.25 * (y + 4716)) + std::floor(30.6001 * (m + 1)) + d + B - 1524.5 + hour / 24.0;
}
inline void dateFromJd(double jd, int& Y, int& M, int& D, double& hour) {
    double Z = std::floor(jd + 0.5), F = jd + 0.5 - Z, A = Z;
    if (Z >= 2299161) { double al = std::floor((Z - 1867216.25) / 36524.25); A = Z + 1 + al - std::floor(al / 4); }
    double B = A + 1524, C = std::floor((B - 122.1) / 365.25), Dd = std::floor(365.25 * C), E = std::floor((B - Dd) / 30.6001);
    D = (int)(B - Dd - std::floor(30.6001 * E));
    M = (int)(E < 14 ? E - 1 : E - 13);
    Y = (int)(M > 2 ? C - 4716 : C - 4715);
    hour = F * 24.0;
}
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 12345) : s(seed * 2685821657736338717ULL + 88172645463325252ULL) { for (int i = 0; i < 4; i++) next(); }
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
    double u() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * u(); }
    double gauss() { double a = std::max(u(), 1e-12), b = u(); return std::sqrt(-2 * std::log(a)) * std::cos(2 * kPI * b); }
    double rayleigh(double sigma) { return sigma * std::sqrt(-2 * std::log(std::max(u(), 1e-12))); }
};
inline uint64_t hashStr(const char* s) { uint64_t h = 1469598103934665603ULL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; } return h; }

// Elementos medios de Standish: a,e,I,L,long.peri,long.nodo + variacion por siglo
struct StdElem { double a, e, I, L, wb, Om, da, de, dI, dL, dwb, dOm; };
static const StdElem kStd[9] = {
    {0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593, 0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081},
    {0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255, 0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418},
    {1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0, 0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0},
    {1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891, 0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343},
    {5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909, -0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106},
    {9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448, -0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794},
    {19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503, -0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589},
    {30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574, 0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664},
    {39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684, -0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482}};
struct Kep { double a, e, inc, Om, w, M; };
Kep stdKep(int idx, double T) {
    const StdElem& s = kStd[idx];
    double a = s.a + s.da * T, e = s.e + s.de * T, I = s.I + s.dI * T, L = s.L + s.dL * T, wb = s.wb + s.dwb * T, Om = s.Om + s.dOm * T;
    return {a, e, I * kDEG, Om * kDEG, (wb - Om) * kDEG, std::fmod(L - wb, 360.0) * kDEG};
}

// ---------------- Aspectos ----------------
inline Look mk(int type, float r1, float g1, float b1, float r2, float g2, float b2, float r3, float g3, float b3,
               float p0 = 1, float p1 = 0, float p2 = 0, float p3 = 0, float lump = 0) {
    Look l; l.type = type;
    l.c1[0] = r1; l.c1[1] = g1; l.c1[2] = b1; l.c2[0] = r2; l.c2[1] = g2; l.c2[2] = b2; l.c3[0] = r3; l.c3[1] = g3; l.c3[2] = b3;
    l.par[0] = p0; l.par[1] = p1; l.par[2] = p2; l.par[3] = p3; l.lump = lump;
    return l;
}
inline Look withAtm(Look l, float r, float g, float b, float thick, float dens) { l.atm[0] = r; l.atm[1] = g; l.atm[2] = b; l.atm[3] = thick; l.atm[4] = dens; return l; }
inline Look withRing(Look l, float in, float out, int type) { l.ringIn = in; l.ringOut = out; l.ringType = type; return l; }

struct Preset { const char* name; Look look; };
inline const std::vector<Preset>& presets() {
    static std::vector<Preset> P;
    if (P.empty()) {
        P.push_back({"Rocoso lunar", mk(0, .56f, .56f, .55f, .30f, .30f, .31f, .78f, .78f, .76f, 3.f, .9f)});
        P.push_back({"Mercurio", mk(0, .50f, .46f, .42f, .30f, .28f, .26f, .72f, .68f, .62f, 3.2f, 1.f)});
        P.push_back({"Marte", withAtm(mk(0, .62f, .30f, .16f, .78f, .50f, .28f, .26f, .15f, .10f, 2.2f, .28f, .035f), .90f, .62f, .45f, .012f, .15f)});
        P.push_back({"Mundo oceánico", withAtm(mk(1, .1f, .3f, .1f, .5f, .4f, .2f, 1, 1, 1), .30f, .55f, 1.f, .025f, .42f)});
        P.push_back({"Gigante gaseoso", withAtm(mk(3, .80f, .66f, .50f, .56f, .37f, .23f, .94f, .89f, .80f, 11.f, .9f, 1.f, 0.f), .80f, .70f, .55f, .008f, .4f)});
        P.push_back({"Gigante dorado", withAtm(mk(3, .88f, .78f, .56f, .72f, .60f, .40f, .95f, .90f, .72f, 9.f, .45f), .85f, .78f, .6f, .008f, .35f)});
        P.push_back({"Gigante de hielo", withAtm(mk(3, .22f, .35f, .85f, .15f, .25f, .70f, .45f, .58f, .95f, 7.f, .35f, .9f, 1.f), .25f, .40f, 1.f, .012f, .6f)});
        P.push_back({"Gigante cian", withAtm(mk(3, .60f, .82f, .86f, .52f, .76f, .82f, .72f, .90f, .92f, 5.f, .12f), .55f, .85f, .95f, .012f, .6f)});
        P.push_back({"Nubes ácidas", withAtm(mk(4, .72f, .62f, .40f, .62f, .50f, .30f, .80f, .72f, .55f, 2.f, .5f), 1.f, .80f, .45f, .040f, .55f)});
        P.push_back({"Neblina naranja", withAtm(mk(4, .86f, .56f, .20f, .70f, .42f, .14f, .93f, .72f, .36f, 2.f, .2f), .95f, .60f, .25f, .050f, 2.0f)});
        P.push_back({"Hielo agrietado", mk(0, .88f, .84f, .76f, .72f, .60f, .48f, .50f, .32f, .22f, 4.f, .12f, 0.f)});
        P.push_back({"Volcánico", mk(0, .88f, .80f, .30f, .86f, .52f, .18f, .96f, .95f, .85f, 3.f, 0.f, 0.f, 1.f)});
        P.push_back({"Desierto", withAtm(mk(0, .78f, .58f, .32f, .62f, .42f, .22f, .90f, .78f, .55f, 2.5f, .4f), .85f, .65f, .45f, .014f, .25f)});
    }
    return P;
}

struct PlanetDef { const char *name, *key; int std; double gmKm, radKm, flat, j2, ra, dec, rotH; unsigned char cr, cg, cb; };
static const PlanetDef kPlanets[9] = {
    {"Mercurio", "mercurio", 0, 22031.78, 2439.7, 0.0, 0.0, 281.01, 61.45, 1407.6, 170, 170, 170},
    {"Venus", "venus", 1, 324858.59, 6051.8, 0.0, 0.0, 272.76, 67.16, -5832.5, 235, 205, 130},
    {"Tierra", "tierra", 2, 398600.436, 6378.137, 1.0 / 298.257, 1.08263e-3, 0.0, 90.0, 23.9345, 90, 160, 255},
    {"Marte", "marte", 3, 42828.37, 3396.19, 0.00589, 1.9605e-3, 317.68, 52.89, 24.6229, 230, 110, 80},
    {"Júpiter", "jupiter", 4, 126686534.0, 71492.0, 0.06487, 1.4736e-2, 268.06, 64.50, 9.925, 225, 180, 130},
    {"Saturno", "saturno", 5, 37931187.0, 60268.0, 0.09796, 1.6298e-2, 40.59, 83.54, 10.656, 230, 210, 150},
    {"Urano", "urano", 6, 5793939.0, 25559.0, 0.0229, 3.34e-3, 257.31, -15.18, -17.24, 130, 215, 225},
    {"Neptuno", "neptuno", 7, 6836529.0, 24764.0, 0.0171, 3.41e-3, 299.36, 43.46, 16.11, 80, 120, 255},
    {"Plutón", "pluton", 8, 869.6, 1188.3, 0.0, 0.0, 132.99, -6.16, -153.29, 205, 175, 150}};

Look planetLook(const std::string& k) {
    const auto& P = presets();
    if (k == "mercurio") return P[1].look;
    if (k == "venus") return P[8].look;
    if (k == "tierra") return P[3].look;
    if (k == "marte") return P[2].look;
    if (k == "jupiter") return P[4].look;
    if (k == "saturno") return withRing(P[5].look, 1.24f, 2.33f, 0);
    if (k == "urano") return withRing(P[7].look, 1.64f, 2.05f, 1);
    if (k == "neptuno") return P[6].look;
    return mk(0, .72f, .60f, .50f, .44f, .30f, .22f, .95f, .92f, .88f, 2.5f, .30f, 0.f);   // Pluton
}

struct MoonDef { const char *name, *key; int parent; double aKm, e, incDeg, gmKm, radKm, lamDeg; float ax, ay, az; unsigned char cr, cg, cb; bool lock; };
static const MoonDef kMoons[] = {
    {"Luna", "luna", 2, 384399, 0.0549, 5.145, 4902.800, 1737.4, 0, 1, 1, 1, 200, 200, 200, true},
    {"Fobos", "fobos", 3, 9376, 0.0151, 1.08, 7.087e-4, 11.1, 40, 1.17f, 1.03f, .82f, 170, 150, 140, true},
    {"Deimos", "deimos", 3, 23463.2, 0.00033, 1.79, 9.62e-5, 6.2, 270, 1.26f, .97f, .82f, 180, 165, 150, true},
    {"Ío", "io", 4, 421800, 0.0041, 0.04, 5959.916, 1821.6, 106.07, 1, 1, 1, 240, 220, 110, true},
    {"Europa", "europa", 4, 671100, 0.0094, 0.47, 3202.739, 1560.8, 175.73, 1, 1, 1, 225, 210, 185, true},
    {"Ganímedes", "ganimedes", 4, 1070400, 0.0013, 0.20, 9887.834, 2631.2, 120.56, 1, 1, 1, 170, 155, 140, true},
    {"Calisto", "calisto", 4, 1882700, 0.0074, 0.19, 7179.289, 2410.3, 84.44, 1, 1, 1, 130, 115, 100, true},
    {"Mimas", "mimas", 5, 185540, 0.0196, 1.57, 2.5026, 198.2, 50, 1, 1, 1, 190, 190, 190, true},
    {"Encélado", "encelado", 5, 238040, 0.0047, 0.01, 7.2093, 252.1, 120, 1, 1, 1, 245, 250, 255, true},
    {"Tetis", "tetis", 5, 294670, 0.0001, 0.17, 41.2097, 531.1, 300, 1, 1, 1, 215, 215, 210, true},
    {"Dione", "dione", 5, 377420, 0.0022, 0.02, 73.1146, 561.4, 10, 1, 1, 1, 205, 205, 200, true},
    {"Rea", "rea", 5, 527070, 0.0010, 0.35, 153.9426, 763.8, 220, 1, 1, 1, 200, 200, 195, true},
    {"Titán", "titan", 5, 1221870, 0.0288, 0.33, 8978.1382, 2574.7, 80, 1, 1, 1, 230, 160, 70, true},
    {"Hiperión", "hiperion", 5, 1500880, 0.1230, 0.43, 0.3727, 135, 160, 1.33f, .99f, .76f, 170, 140, 115, false},
    {"Jápeto", "japeto", 5, 3560820, 0.0286, 15.47, 120.5038, 734.5, 200, 1, 1, 1, 190, 180, 165, true},
    {"Miranda", "miranda", 6, 129900, 0.0013, 175.66, 4.4, 235.8, 20, 1, 1, 1, 190, 190, 190, true},
    {"Ariel", "ariel", 6, 190900, 0.0012, 179.96, 86.4, 578.9, 140, 1, 1, 1, 200, 200, 198, true},
    {"Umbriel", "umbriel", 6, 266000, 0.0039, 179.87, 81.5, 584.7, 260, 1, 1, 1, 130, 130, 130, true},
    {"Titania", "titania", 6, 436300, 0.0011, 179.92, 228.2, 788.9, 60, 1, 1, 1, 175, 165, 158, true},
    {"Oberón", "oberon", 6, 583500, 0.0014, 179.93, 192.4, 761.4, 200, 1, 1, 1, 160, 148, 138, true},
    {"Tritón", "triton", 7, 354759, 0.000016, 156.885, 1427.6, 1353.4, 30, 1, 1, 1, 235, 215, 205, true},
    {"Proteo", "proteo", 7, 117647, 0.0005, 0.08, 3.3, 210, 100, 1.04f, .99f, .96f, 150, 150, 150, true},
    {"Nereida", "nereida", 7, 5513787, 0.7507, 29.0, 2.07, 170, 250, 1, 1, 1, 180, 180, 180, true},
    {"Caronte", "caronte", 8, 19596, 0.0002, 179.92, 105.88, 606, 40, 1, 1, 1, 175, 170, 168, true},
    {"Nix", "nix", 8, 48694, 0.0020, 179.8, 0.0030, 20, 120, 1.25f, .83f, .78f, 200, 200, 200, true},
    {"Hidra", "hidra", 8, 64738, 0.0059, 179.8, 0.0032, 20, 200, 1.35f, .95f, .70f, 205, 205, 205, true}};
static const int kNumMoons = (int)(sizeof(kMoons) / sizeof(kMoons[0]));

Look moonLook(const std::string& k) {
    if (k == "luna") return mk(0, .62f, .61f, .60f, .27f, .27f, .28f, .86f, .85f, .83f, 3.f, 1.f);
    if (k == "fobos" || k == "deimos") return mk(0, .32f, .28f, .26f, .20f, .18f, .17f, .44f, .40f, .36f, 4.f, .8f, 0, 0, 1.f);
    if (k == "io") return mk(0, .88f, .80f, .30f, .86f, .52f, .18f, .96f, .95f, .85f, 3.f, 0.f, 0.f, 1.f);
    if (k == "europa") return mk(0, .88f, .84f, .76f, .72f, .60f, .48f, .50f, .32f, .22f, 4.f, .12f, 0.f);
    if (k == "ganimedes") return mk(0, .52f, .47f, .42f, .34f, .31f, .29f, .78f, .76f, .74f, 3.f, .7f);
    if (k == "calisto") return mk(0, .36f, .31f, .27f, .26f, .23f, .21f, .60f, .57f, .54f, 5.f, 1.2f);
    if (k == "mimas") return mk(0, .62f, .62f, .62f, .45f, .45f, .46f, .80f, .80f, .80f, 3.5f, 1.3f);
    if (k == "encelado") return mk(0, .96f, .97f, .98f, .88f, .92f, .96f, 1, 1, 1, 3.f, .2f, 0.f);
    if (k == "tetis") return mk(0, .78f, .78f, .76f, .62f, .62f, .60f, .90f, .90f, .88f, 3.f, .8f);
    if (k == "dione") return mk(0, .74f, .74f, .72f, .55f, .55f, .54f, .88f, .88f, .86f, 3.f, .8f);
    if (k == "rea") return mk(0, .70f, .70f, .68f, .52f, .52f, .50f, .86f, .86f, .84f, 3.f, .9f);
    if (k == "titan") return withAtm(mk(4, .86f, .56f, .20f, .70f, .42f, .14f, .93f, .72f, .36f, 2.f, .2f), .95f, .60f, .25f, .05f, 2.0f);
    if (k == "hiperion") return mk(0, .60f, .50f, .42f, .40f, .32f, .26f, .72f, .62f, .52f, 4.f, .9f, 0, 0, 1.f);
    if (k == "japeto") return mk(0, .12f, .10f, .08f, .86f, .86f, .82f, .55f, .52f, .48f, 3.f, .8f, 0.f, 2.f);
    if (k == "miranda") return mk(0, .62f, .62f, .62f, .40f, .40f, .41f, .82f, .82f, .82f, 4.f, .9f);
    if (k == "ariel") return mk(0, .66f, .66f, .65f, .45f, .45f, .45f, .84f, .84f, .83f, 3.f, .6f);
    if (k == "umbriel") return mk(0, .36f, .36f, .36f, .26f, .26f, .27f, .52f, .52f, .52f, 3.f, .8f);
    if (k == "titania") return mk(0, .55f, .52f, .50f, .38f, .35f, .33f, .72f, .70f, .68f, 3.f, .7f);
    if (k == "oberon") return mk(0, .50f, .46f, .43f, .34f, .31f, .29f, .66f, .62f, .59f, 3.f, .9f);
    if (k == "triton") return mk(0, .88f, .80f, .76f, .70f, .55f, .50f, .95f, .92f, .90f, 3.f, .3f, .08f);
    if (k == "proteo" || k == "nereida") return mk(0, .40f, .40f, .40f, .28f, .28f, .28f, .55f, .55f, .55f, 4.f, .9f, 0, 0, 1.f);
    if (k == "caronte") return mk(0, .62f, .60f, .60f, .40f, .37f, .37f, .80f, .78f, .78f, 3.f, .7f);
    if (k == "nix" || k == "hidra") return mk(0, .75f, .75f, .75f, .55f, .55f, .55f, .90f, .90f, .90f, 4.f, .5f, 0, 0, 1.f);
    return presets()[0].look;
}

struct SmallDef { const char *name, *key; double a, e, inc, Om, w, M, gmKm, radKm, ra, dec, rotH; float ax, ay, az; unsigned char cr, cg, cb; bool massive; };
static const SmallDef kSmall[] = {
    {"Ceres", "ceres", 2.7670, 0.0785, 10.588, 80.27, 73.60, 193, 62.6284, 469.7, 291.43, 66.76, 9.074, 1, 1, .95f, 190, 190, 185, true},
    {"Vesta", "vesta", 2.3615, 0.0887, 7.14, 103.81, 150.73, 71, 17.288, 262.7, 309.03, 42.23, 5.342, 1.09f, 1.06f, .85f, 205, 190, 165, true},
    {"Palas", "palas", 2.7716, 0.2305, 34.84, 173.03, 310.20, 255, 13.6, 256, 33.0, -3.0, 7.813, 1.06f, 1.0f, .94f, 175, 175, 170, true},
    {"Haumea", "haumea", 43.22, 0.1953, 28.21, 121.9, 240.2, 215, 267.0, 816, 282.6, -10.6, 3.9154, 1.42f, 1.04f, .70f, 225, 225, 230, true},
    {"Makemake", "makemake", 45.51, 0.1550, 28.98, 79.38, 294.8, 170, 207.0, 715, 0.0, 90.0, 22.8, 1, 1, 1, 200, 140, 110, true},
    {"Eris", "eris", 67.78, 0.4407, 44.04, 35.95, 151.64, 211, 1108.0, 1163, 0.0, 90.0, 25.9, 1, 1, 1, 235, 235, 230, true},
    {"Sedna", "sedna", 506.0, 0.8496, 11.93, 144.5, 311.4, 358.4, 0.0, 500, 0.0, 90.0, 10.3, 1, 1, 1, 200, 80, 60, false}};
static const int kNumSmall = (int)(sizeof(kSmall) / sizeof(kSmall[0]));
Look smallLook(const std::string& k) {
    if (k == "ceres") return mk(0, .40f, .38f, .36f, .28f, .27f, .26f, .80f, .80f, .78f, 4.f, .9f);
    if (k == "vesta") return mk(0, .62f, .58f, .52f, .40f, .37f, .33f, .80f, .76f, .70f, 4.f, .8f, 0, 0, .4f);
    if (k == "palas") return mk(0, .50f, .50f, .48f, .36f, .36f, .35f, .70f, .70f, .68f, 4.f, .8f, 0, 0, .5f);
    if (k == "haumea") return mk(0, .88f, .88f, .88f, .75f, .75f, .78f, .55f, .35f, .30f, 3.f, .15f, 0.f);
    if (k == "makemake") return mk(0, .66f, .44f, .32f, .52f, .34f, .25f, .85f, .80f, .75f, 3.f, .2f, 0.f);
    if (k == "eris") return mk(0, .88f, .87f, .85f, .78f, .77f, .75f, .95f, .95f, .93f, 3.f, .2f);
    return mk(0, .55f, .20f, .12f, .40f, .14f, .09f, .70f, .30f, .20f, 3.f, .25f);   // Sedna
}

struct CometDef { const char *name, *key; double a, e, inc, Om, w, Tp; };
static const CometDef kComets[] = {
    {"Halley", "halley", 17.8341, 0.96714, 162.263, 58.42, 111.33, 2446470.95},
    {"Encke", "encke", 2.2152, 0.8483, 11.78, 334.57, 186.54, 2460240.6},
    {"67P/Churyumov-Gerasimenko", "67p", 3.4630, 0.6405, 7.04, 36.33, 22.15, 2459520.6},
    {"Hale-Bopp", "halebopp", 186.0, 0.99508, 89.43, 282.47, 130.59, 2450539.6}};
static const int kNumComets = (int)(sizeof(kComets) / sizeof(kComets[0]));

// ---------------- Teoria lunar (Meeus, cap. 47, terminos principales) ----------------
void moonGeoEcliptic(double jd, V3& r) {
    static const int TL[16][6] = {   // D, M, M', F, sum_l (1e-6 deg), sum_r (1e-3 km)
        {0, 0, 1, 0, 6288774, -20905355}, {2, 0, -1, 0, 1274027, -3699111}, {2, 0, 0, 0, 658314, -2955968}, {0, 0, 2, 0, 213618, -569925},
        {0, 1, 0, 0, -185116, 48888}, {0, 0, 0, 2, -114332, -3149}, {2, 0, -2, 0, 58793, 246158}, {2, -1, -1, 0, 57066, -152138},
        {2, 0, 1, 0, 53322, -170733}, {2, -1, 0, 0, 45758, -204586}, {0, 1, -1, 0, -40923, -129620}, {1, 0, 0, 0, -34720, 108743},
        {0, 1, 1, 0, -30383, 104755}, {2, 0, 0, -2, 15327, 10321}, {0, 0, 1, 2, -12528, 0}, {0, 0, 1, -2, 10980, 79661}};
    static const int TB[12][5] = {   // D, M, M', F, sum_b (1e-6 deg)
        {0, 0, 0, 1, 5128122}, {0, 0, 1, 1, 280602}, {0, 0, 1, -1, 277693}, {2, 0, 0, -1, 173237}, {2, 0, -1, 1, 55413}, {2, 0, -1, -1, 46271},
        {2, 0, 0, 1, 32573}, {0, 0, 2, 1, 17198}, {2, 0, 1, -1, 9266}, {0, 0, 2, -1, 8822}, {2, -1, 0, -1, 8216}, {2, 0, -2, -1, 4324}};
    double T = (jd - 2451545.0) / 36525.0;
    double Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T * T;
    double D = (297.8501921 + 445267.1114034 * T - 0.0018819 * T * T) * kDEG;
    double M = (357.5291092 + 35999.0502909 * T - 0.0001536 * T * T) * kDEG;
    double Mp = (134.9633964 + 477198.8675055 * T + 0.0087414 * T * T) * kDEG;
    double F = (93.2720950 + 483202.0175233 * T - 0.0036539 * T * T) * kDEG;
    double E = 1 - 0.002516 * T - 0.0000074 * T * T;
    double A1 = (119.75 + 131.849 * T) * kDEG, A2 = (53.09 + 479264.290 * T) * kDEG, A3 = (313.45 + 481266.484 * T) * kDEG, Lr = Lp * kDEG;
    double sl = 0, sr = 0, sb = 0;
    for (auto& t : TL) {
        double arg = t[0] * D + t[1] * M + t[2] * Mp + t[3] * F, f = (t[1] == 0) ? 1.0 : (std::abs(t[1]) == 1 ? E : E * E);
        sl += t[4] * f * std::sin(arg); sr += t[5] * f * std::cos(arg);
    }
    for (auto& t : TB) {
        double arg = t[0] * D + t[1] * M + t[2] * Mp + t[3] * F, f = (t[1] == 0) ? 1.0 : (std::abs(t[1]) == 1 ? E : E * E);
        sb += t[4] * f * std::sin(arg);
    }
    sl += 3958 * std::sin(A1) + 1962 * std::sin(Lr - F) + 318 * std::sin(A2);
    sb += -2235 * std::sin(Lr) + 382 * std::sin(A3) + 175 * std::sin(A1 - F) + 175 * std::sin(A1 + F) + 127 * std::sin(Lr - Mp) - 115 * std::sin(Lr + Mp);
    double lon = (Lp + sl / 1e6 - 1.3969713 * T) * kDEG;   // al equinoccio J2000
    double lat = (sb / 1e6) * kDEG;
    double dist = (385000.56 + sr / 1000.0) / AU_KM;
    r = {dist * std::cos(lat) * std::cos(lon), dist * std::cos(lat) * std::sin(lon), dist * std::sin(lat)};
}

// ---------------- Construccion del modelo ----------------
struct BuildOpts { int nMain = 6000, nTroj = 1400, nHilda = 300, nKuiper = 2800, nNeo = 150, nCentaur = 40; };

void snapshotElements(Sim& S) {
    for (size_t i = 1; i < S.b.size(); i++) {
        Body& bd = S.b[i];
        if (!bd.alive || bd.kind == K_STAR) continue;
        V3 r, v, c; double mu; Mat3 F;
        if (!S.relState((int)i, r, v, mu, c, F)) continue;
        Elems el = elementsFromState(mulMT(F, r), mulMT(F, v), mu);
        if (el.bound) { bd.a0 = el.a; bd.e0 = el.e; bd.i0 = el.inc; }
        if (bd.key == "luna") { bd.a0 = 384399.0 / AU_KM; bd.e0 = 0.0549; bd.i0 = 5.145 * kDEG; }
    }
}

void buildSolarSystem(Sim& S, double jd, const BuildOpts& o) {
    S = Sim();
    S.jd0 = jd;
    const double T = (jd - 2451545.0) / 36525.0;
    const double gmSun = gmConv(132712440041.94);
    Rng rg(2026);
    auto addBody = [&](const char* name, const char* key, BodyKind k) -> int {
        Body b; b.name = name; b.key = key; b.kind = k; S.b.push_back(b); return (int)S.b.size() - 1;
    };
    auto finishBody = [&](int i) { Body& b = S.b[i]; b.gm0 = b.gm; b.radius0 = b.radiusKm; b.flat0 = b.flat; b.spin0 = b.spin; b.spinRate0 = b.spinRate; };

    // --- Sol
    int sun = addBody("Sol", "sol", K_STAR);
    {
        Body& b = S.b[sun];
        b.gm = gmSun; b.radiusKm = 695700.0; b.spin = poleFromRaDec(286.13, 63.87); b.spinRate = 2 * kPI / 25.38;
        b.look = mk(5, 1.f, .85f, .5f, 1.f, .55f, .15f, 1.f, .95f, .8f); b.col = {255, 220, 120, 255};
        finishBody(sun);
        Prim pr; pr.body = sun; pr.gm = gmSun; S.p.push_back(pr);
        b.prim = 0;
    }
    // --- Planetas + lunas (cada sistema = baricentro integrado en el campo del Sol)
    std::vector<V3> hr, hv;   // estado heliocentrico de cada primario (alineado con S.p)
    hr.push_back(V3{}); hv.push_back(V3{});
    int earthBody = -1;
    for (const PlanetDef& d : kPlanets) {
        int bi = addBody(d.name, d.key, K_PLANET);
        double gmP = gmConv(d.gmKm), gmSys = gmP;
        for (int m = 0; m < kNumMoons; m++) if (kMoons[m].parent == d.std) gmSys += gmConv(kMoons[m].gmKm);
        {
            Body& b = S.b[bi];
            b.gm = gmP; b.radiusKm = d.radKm; b.flat = d.flat; b.j2 = d.j2;
            b.spin = poleFromRaDec(d.ra, d.dec); b.spinRate = 2 * kPI / (d.rotH / 24.0);
            b.spinPhase = rg.u() * 2 * kPI; b.col = {d.cr, d.cg, d.cb, 255};
            b.look = planetLook(d.key); b.dwarf = (d.std == 8); b.rings = b.look.ringOut > 0;
            finishBody(bi);
        }
        if (d.std == 2) earthBody = bi;
        Kep k = stdKep(d.std, T);
        Prim pr; pr.body = bi; pr.gm = gmSys;
        V3 r, v; stateFromElements(gmSun + gmSys, k.a, k.e, k.inc, k.Om, k.w, k.M, r, v);
        pr.x = r; pr.v = v;
        int pidx = (int)S.p.size();
        // lunas
        Mat3 F = eqFrame(S.b[bi].spin);
        for (int m = 0; m < kNumMoons; m++) {
            const MoonDef& md = kMoons[m];
            if (md.parent != d.std) continue;
            int mi = addBody(md.name, md.key, K_MOON);
            Body& mb = S.b[mi];
            mb.parent = bi; mb.prim = pidx; mb.moonSlot = (int)pr.moons.size();
            mb.gm = gmConv(md.gmKm); mb.radiusKm = md.radKm; mb.axes = {md.ax, md.ay, md.az};
            mb.tidalLock = md.lock; mb.refEcliptic = (std::string(md.key) == "luna"); mb.col = {md.cr, md.cg, md.cb, 255};
            mb.look = moonLook(md.key);
            if (!md.lock) { mb.spin = vnorm(V3{0.3, -0.5, 0.8}); mb.spinRate = 4.0; }
            finishBody(mi);
            V3 rr, vv;
            if (mb.refEcliptic) {
                V3 r0, r1, r2; double h = 0.02;
                moonGeoEcliptic(jd - h, r0); moonGeoEcliptic(jd, r1); moonGeoEcliptic(jd + h, r2);
                rr = r1; vv = (r2 - r0) * (1.0 / (2 * h));
            } else {
                Rng rm(hashStr(md.key));
                double Om = rm.u() * 2 * kPI, w = rm.u() * 2 * kPI, M = md.lamDeg * kDEG - Om - w;
                V3 rl, vl;
                stateFromElements(gmP + mb.gm, md.aKm / AU_KM, md.e, md.incDeg * kDEG, Om, w, M, rl, vl);
                rr = mulM(F, rl); vv = mulM(F, vl);
            }
            pr.moons.push_back(mi); pr.r.push_back(rr); pr.w.push_back(vv);
        }
        // el baricentro del sistema ya es pr.x; el planeta se desplaza en syncBodies()
        S.p.push_back(pr); S.b[bi].prim = pidx;
        hr.push_back(pr.x); hv.push_back(pr.v);
    }
    // --- Enanos y asteroides grandes (heliocentricos)
    for (int i = 0; i < kNumSmall; i++) {
        const SmallDef& d = kSmall[i];
        int bi = addBody(d.name, d.key, K_SMALL);
        Body& b = S.b[bi];
        b.gm = gmConv(d.gmKm); b.radiusKm = d.radKm; b.axes = {d.ax, d.ay, d.az};
        b.massless = !d.massive; b.dwarf = (d.a > 30 || std::string(d.key) == "ceres");
        b.spin = poleFromRaDec(d.ra, d.dec); b.spinRate = 2 * kPI / (d.rotH / 24.0); b.spinPhase = rg.u() * 2 * kPI;
        b.col = {d.cr, d.cg, d.cb, 255}; b.look = smallLook(d.key);
        finishBody(bi);
        Prim pr; pr.body = bi; pr.massless = b.massless; pr.gm = b.massless ? 0.0 : b.gm;
        V3 r, v; stateFromElements(gmSun + pr.gm, d.a, d.e, d.inc * kDEG, d.Om * kDEG, d.w * kDEG, d.M * kDEG, r, v);
        pr.x = r; pr.v = v;
        b.prim = (int)S.p.size();
        S.p.push_back(pr); hr.push_back(r); hv.push_back(v);
    }
    // --- Pasar a coordenadas baricentricas (momento total nulo)
    {
        double Mt = gmSun; V3 Rc, Vc;
        for (size_t k = 1; k < S.p.size(); k++) if (!S.p[k].massless) { Mt += S.p[k].gm; Rc += hr[k] * S.p[k].gm; Vc += hv[k] * S.p[k].gm; }
        Rc = Rc * (1.0 / Mt); Vc = Vc * (1.0 / Mt);
        for (size_t k = 1; k < S.p.size(); k++) { S.p[k].x = hr[k] - Rc; S.p[k].v = hv[k] - Vc; }
        S.p[0].x = V3{} - Rc; S.p[0].v = V3{} - Vc;
    }
    V3 sunPos = S.p[0].x, sunVel = S.p[0].v;
    // --- Cometas (sin masa, del perihelio al estado actual)
    for (int i = 0; i < kNumComets; i++) {
        const CometDef& d = kComets[i];
        int bi = addBody(d.name, d.key, K_COMET);
        Body& b = S.b[bi];
        b.massless = true; b.radiusKm = 6.0; b.axes = {1.3, 1.0, 0.9}; b.col = {150, 220, 255, 255};
        b.look = mk(0, .10f, .09f, .08f, .06f, .06f, .06f, .22f, .21f, .20f, 4.f, .8f, 0, 0, 1.f);
        b.spin = vnorm(V3{0.2 + 0.1 * i, 0.5, 0.8}); b.spinRate = 2 * kPI / 0.6; b.spinPhase = rg.u() * 2 * kPI;
        finishBody(bi);
        double n = std::sqrt(gmSun / (d.a * d.a * d.a)), M = std::fmod(n * (jd - d.Tp), 2 * kPI);
        Prim pr; pr.body = bi; pr.massless = true; pr.gm = 0;
        V3 r, v; stateFromElements(gmSun, d.a, d.e, d.inc * kDEG, d.Om * kDEG, d.w * kDEG, M, r, v);
        pr.x = sunPos + r; pr.v = sunVel + v;
        b.prim = (int)S.p.size(); S.p.push_back(pr);
    }
    // --- Enjambre de particulas de prueba
    Rng rs(777);
    S.sw.reserve(o.nMain + o.nTroj + o.nHilda + o.nKuiper + o.nNeo + o.nCentaur);
    auto addP = [&](double a, double e, double inc, double Om, double w, double M, int type, double size) {
        if (a * (1 - e) < 0.08 || e >= 0.97) return false;
        V3 r, v; stateFromElements(gmSun, a, e, inc, Om, w, M, r, v);
        Particle q; q.x = sunPos + r; q.v = sunVel + v; q.size = (float)size; q.type = (unsigned char)type;
        q.shade = (unsigned char)rs.range(110, 255); q.alive = true; S.sw.push_back(q);
        return true;
    };
    auto sz = [&]() { return 0.7 + 2.6 * std::pow(rs.u(), 7.0); };
    auto kirk = [](double a) {
        static const double g[6][2] = {{2.065, 0.012}, {2.502, 0.035}, {2.706, 0.010}, {2.825, 0.020}, {2.958, 0.015}, {3.279, 0.035}};
        for (auto& x : g) if (std::fabs(a - x[0]) < x[1]) return true;
        return false;
    };
    const double LJ = std::fmod(kStd[4].L + kStd[4].dL * T, 360.0) * kDEG;
    for (int n = 0; n < o.nMain;) {
        double a = rs.range(2.1, 3.3); if (kirk(a)) continue;
        double e = std::min(rs.rayleigh(0.08), 0.35), inc = std::min(rs.rayleigh(0.1), 0.5);
        if (addP(a, e, inc, rs.u() * 2 * kPI, rs.u() * 2 * kPI, rs.u() * 2 * kPI, 0, sz())) n++;
    }
    for (int n = 0; n < o.nHilda;) {
        double a = rs.gauss() * 0.07 + 3.97, e = std::min(std::fabs(rs.gauss() * 0.07 + 0.15), 0.35);
        if (addP(a, e, std::min(rs.rayleigh(0.1), 0.4), rs.u() * 2 * kPI, rs.u() * 2 * kPI, rs.u() * 2 * kPI, 2, sz())) n++;
    }
    for (int n = 0; n < o.nTroj;) {
        double side = (n & 1) ? 1.0 : -1.0, a = 5.2029 + rs.gauss() * 0.12, e = std::min(rs.rayleigh(0.05), 0.2), inc = std::min(rs.rayleigh(0.17), 0.5);
        double Om = rs.u() * 2 * kPI, w = rs.u() * 2 * kPI, lam = LJ + side * (kPI / 3) + rs.gauss() * 0.17;
        if (addP(a, e, inc, Om, w, lam - Om - w, 1, sz())) n++;
    }
    for (int n = 0; n < o.nKuiper;) {
        double a, e, inc; double c = rs.u();
        if (c < 0.62) { a = std::min(std::max(rs.gauss() * 2.0 + 44.0, 41.0), 48.5); e = std::min(rs.rayleigh(0.06), 0.2); inc = rs.rayleigh(c < 0.35 ? 0.07 : 0.22); }
        else if (c < 0.85) { a = 39.4 + rs.gauss() * 0.15; e = rs.range(0.08, 0.30); inc = rs.rayleigh(0.17); }
        else { double q = rs.range(35, 46); a = rs.range(50, 110); e = 1 - q / a; inc = rs.rayleigh(0.2); }
        if (addP(a, e, std::min(inc, 0.9), rs.u() * 2 * kPI, rs.u() * 2 * kPI, rs.u() * 2 * kPI, 3, sz() * 0.9)) n++;
    }
    for (int n = 0; n < o.nCentaur;) {
        double a = rs.range(8, 28), e = rs.range(0.05, 0.45);
        if (addP(a, e, std::min(rs.rayleigh(0.25), 1.0), rs.u() * 2 * kPI, rs.u() * 2 * kPI, rs.u() * 2 * kPI, 3, sz())) n++;
    }
    for (int n = 0; n < o.nNeo;) {
        double a = rs.range(0.85, 2.4), e = rs.range(0.1, 0.7), q = a * (1 - e);
        if (q < 0.3 || q > 1.3) continue;
        if (addP(a, e, std::min(rs.rayleigh(0.18), 0.7), rs.u() * 2 * kPI, rs.u() * 2 * kPI, rs.u() * 2 * kPI, 4, sz() * 1.2)) n++;
    }
    // --- Orientacion de la Tierra: tiempo sideral de Greenwich (el dia se ve en su hora real)
    {
        double gmst = std::fmod(280.46061837 + 360.98564736629 * (jd - 2451545.0), 360.0) * kDEG;
        S.b[earthBody].spinPhase = std::fmod(kPI + gmst + 8 * kPI, 2 * kPI);
    }
    S.Hmax = 0.5; S.moonDiv = 60.0;
    S.recomputeMasses();
    S.syncBodies();
    snapshotElements(S);
    for (auto& b : S.b) b.dispPhase = b.spinPhase;
}

// ---------------- Edicion: crear / eliminar cuerpos ----------------
int addUserPlanet(Sim& S, double aAU, int presetIdx, Rng& rg) {
    const auto& P = presets();
    Body b; b.kind = K_PLANET; b.userCreated = true;
    char nm[48]; int cnt = 1; for (auto& x : S.b) if (x.userCreated && x.kind == K_PLANET) cnt++;
    std::snprintf(nm, sizeof nm, "Planeta X-%d", cnt);
    b.name = nm; b.key = nm;
    b.gm = gmConv(398600.436); b.radiusKm = 6378.0; b.spin = vnorm(V3{rg.range(-.4, .4), rg.range(-.4, .4), 1.0}); b.spinRate = 2 * kPI;
    b.spinPhase = rg.u() * 2 * kPI; b.col = {255, 140, 220, 255}; b.look = P[presetIdx % P.size()].look;
    b.gm0 = b.gm; b.radius0 = b.radiusKm; b.spin0 = b.spin; b.spinRate0 = b.spinRate; b.dispPhase = b.spinPhase;
    Prim pr; pr.gm = b.gm;
    V3 r, v; stateFromElements(S.b[0].gm + b.gm, aAU, 0.02, rg.range(0, 0.05), rg.u() * 2 * kPI, rg.u() * 2 * kPI, rg.u() * 2 * kPI, r, v);
    pr.x = S.p[0].x + r; pr.v = S.p[0].v + v;
    S.b.push_back(b); int bi = (int)S.b.size() - 1;
    pr.body = bi; S.b[bi].prim = (int)S.p.size(); S.p.push_back(pr);
    S.recomputeMasses(); S.syncBodies();
    S.b[bi].a0 = aAU; S.b[bi].e0 = 0.02; S.b[bi].i0 = 0.0;
    return bi;
}
int addUserMoon(Sim& S, int planet, Rng& rg) {
    Body& pl = S.b[planet];
    if (pl.kind == K_MOON) return -1;
    Prim& pr = S.p[pl.prim];
    double Rp = pl.radiusKm / AU_KM, rHill = vlen(pr.x - S.p[0].x) * std::cbrt(pr.gm / (3 * S.b[0].gm));
    double a = std::min(std::max(8.0 * Rp, 0.0), 0.25 * rHill);
    a = std::max(a, 2.6 * Rp);
    Body b; b.kind = K_MOON; b.userCreated = true; b.parent = planet; b.prim = pl.prim; b.moonSlot = (int)pr.moons.size();
    char nm[48]; std::snprintf(nm, sizeof nm, "Luna X-%d", (int)pr.moons.size() + 1);
    b.name = nm; b.key = nm; b.gm = gmConv(100.0); b.radiusKm = 900; b.tidalLock = true; b.refEcliptic = false;
    b.look = presets()[rg.next() % 3 == 0 ? 10 : 0].look; b.col = {230, 230, 255, 255};
    b.gm0 = b.gm; b.radius0 = b.radiusKm;
    S.b.push_back(b); int bi = (int)S.b.size() - 1;
    Mat3 F = eqFrame(pl.spin);
    V3 rl, vl; stateFromElements(pl.gm + b.gm, a, 0.01, rg.range(0, 0.05), rg.u() * 2 * kPI, rg.u() * 2 * kPI, rg.u() * 2 * kPI, rl, vl);
    pr.moons.push_back(bi); pr.r.push_back(mulM(F, rl)); pr.w.push_back(mulM(F, vl));
    S.recomputeMasses(); S.syncBodies();
    S.b[bi].a0 = a; S.b[bi].e0 = 0.01; S.b[bi].i0 = 0;
    return bi;
}
int addUserComet(Sim& S, Rng& rg) {
    Body b; b.kind = K_COMET; b.userCreated = true; b.massless = true; b.radiusKm = 5; b.axes = {1.3, 1, .9};
    char nm[48]; int cnt = 1; for (auto& x : S.b) if (x.userCreated && x.kind == K_COMET) cnt++;
    std::snprintf(nm, sizeof nm, "Cometa X-%d", cnt);
    b.name = nm; b.key = nm; b.col = {150, 220, 255, 255};
    b.look = mk(0, .10f, .09f, .08f, .06f, .06f, .06f, .22f, .21f, .20f, 4.f, .8f, 0, 0, 1.f);
    b.spin = vnorm(V3{.2, .5, .8}); b.spinRate = 2 * kPI / 0.6;
    Prim pr; pr.massless = true;
    double a = rg.range(3.0, 9.0), e = rg.range(0.82, 0.93);
    V3 r, v; stateFromElements(S.b[0].gm, a, e, rg.range(0, 3.0), rg.u() * 2 * kPI, rg.u() * 2 * kPI, rg.range(2.0, 4.0), r, v);
    pr.x = S.p[0].x + r; pr.v = S.p[0].v + v;
    S.b.push_back(b); int bi = (int)S.b.size() - 1;
    pr.body = bi; S.b[bi].prim = (int)S.p.size(); S.p.push_back(pr);
    S.recomputeMasses(); S.syncBodies();
    S.b[bi].a0 = a; S.b[bi].e0 = e; S.b[bi].i0 = 0;
    return bi;
}
void removeBody(Sim& S, int i) {
    if (i <= 0 || i >= (int)S.b.size()) return;
    Body& bd = S.b[i];
    bd.alive = false;
    if (bd.kind == K_MOON) { /* queda inerte en su ranura */ }
    else {
        S.p[bd.prim].alive = false;
        for (int mi : S.p[bd.prim].moons) S.b[mi].alive = false;
    }
    S.recomputeMasses(); S.syncBodies();
}

// ============================================================================
//  SHADERS GLSL 330 (procedurales: planetas, atmosferas, anillos, Sol, cielo)
// ============================================================================
static const std::string kGlslNoise =
    R"GLSL(float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, vec3(p.z, p.y, p.x) + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, vec3(p.y, p.x, p.z) + 33.33);
    return fract(vec3((p.x + p.y) * p.z, (p.x + p.x) * p.y, (p.y + p.x) * p.x));
}
float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash13(i);
    float b = hash13(i + vec3(1.0, 0.0, 0.0));
    float c = hash13(i + vec3(0.0, 1.0, 0.0));
    float d = hash13(i + vec3(1.0, 1.0, 0.0));
    float e = hash13(i + vec3(0.0, 0.0, 1.0));
    float g = hash13(i + vec3(1.0, 0.0, 1.0));
    float h = hash13(i + vec3(0.0, 1.0, 1.0));
    float k = hash13(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), mix(mix(e, g, f.x), mix(h, k, f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * vnoise(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}
)GLSL";

static const std::string kGlslBodyVS =
    R"GLSL(in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
uniform int uType;
uniform float uLump;
uniform float uSeed;
out vec3 vPos;
out vec3 vNrm;
out vec3 vObj;
void main() {
    vec3 p = vertexPosition;
    if (uType < 10 && uLump > 0.0) {
        float n = fbm(normalize(p) * 2.3 + vec3(uSeed)) - 0.48;
        p *= 1.0 + uLump * 0.55 * n;
    }
    vec4 wp = matModel * vec4(p, 1.0);
    vPos = wp.xyz;
    vNrm = normalize(mat3(matNormal) * vertexNormal);
    vObj = vertexPosition;
    gl_Position = mvp * vec4(p, 1.0);
}
)GLSL";

static const std::string kGlslBodyFS =
    R"GLSL(in vec3 vPos;
in vec3 vNrm;
in vec3 vObj;
out vec4 finalColor;

uniform int uType;
uniform vec3 uCol1;
uniform vec3 uCol2;
uniform vec3 uCol3;
uniform vec4 uPar;
uniform vec4 uAtm;
uniform float uAtmDen;
uniform vec3 uCenter;
uniform float uRadius;
uniform vec3 uSun;
uniform float uSunI;
uniform float uSunAng;
uniform float uAmb;
uniform float uTime;
uniform float uClock;
uniform float uLump;
uniform vec3 uRingN;
uniform vec3 uRingR;
uniform vec4 uOcc[6];
uniform int uOccN;
uniform float uSeed;
uniform float uFlat;

const float PI = 3.14159265;

float shadowAt(vec3 P, vec3 L) {
    float s = 1.0;
    for (int i = 0; i < 6; i++) {
        if (i >= uOccN) break;
        vec3 c = uOcc[i].xyz;
        float r = uOcc[i].w;
        float t = dot(c - P, L);
        if (t > 0.0) {
            float d = length(P + L * t - c);
            float pen = max(t * uSunAng, 1e-6);
            s *= smoothstep(r - pen, r + pen, d);
        }
    }
    return s;
}

float ringDensity(float r) {
    float d = 0.0;
    if (uRingR.z < 0.5) {
        d += smoothstep(1.24, 1.27, r) * (1.0 - smoothstep(1.51, 1.54, r)) * 0.22;
        float bb = smoothstep(1.52, 1.57, r) * (1.0 - smoothstep(1.94, 1.96, r));
        d += bb * (0.78 + 0.22 * sin(r * 210.0) * sin(r * 77.0));
        d += smoothstep(2.03, 2.05, r) * (1.0 - smoothstep(2.265, 2.28, r)) * 0.55;
        d *= 1.0 - 0.92 * (1.0 - smoothstep(0.0, 0.006, abs(r - 2.214)));
        d += 0.35 * (1.0 - smoothstep(0.0, 0.006, abs(r - 2.325)));
        d *= 0.75 + 0.25 * vnoise(vec3(r * 260.0, 1.0, 2.0));
    } else {
        float x = 0.0;
        x += 1.0 - smoothstep(0.0, 0.010, abs(r - 1.64));
        x += 1.0 - smoothstep(0.0, 0.010, abs(r - 1.68));
        x += 1.0 - smoothstep(0.0, 0.010, abs(r - 1.73));
        x += 1.0 - smoothstep(0.0, 0.010, abs(r - 1.78));
        x += 1.0 - smoothstep(0.0, 0.012, abs(r - 1.86));
        x += 1.0 - smoothstep(0.0, 0.012, abs(r - 1.92));
        x += 1.6 * (1.0 - smoothstep(0.0, 0.020, abs(r - 2.0)));
        d = x * 0.35;
    }
    return clamp(d, 0.0, 1.0);
}

vec3 sq(vec3 v) {
    return v + uRingN * (dot(v, uRingN) * uFlat);
}

// dispersion simple en una esfera (estirada a lo largo del polo si el planeta es oblato) con atmosfera exponencial
// rgb = luz dispersada, a = transmitancia
vec4 atmosphere(vec3 ro, vec3 rd, vec3 L) {
    float Rp = uRadius;
    float Ra = uRadius * (1.0 + uAtm.w);
    vec3 oc = sq(ro - uCenter);
    vec3 dd = sq(rd);
    float a = dot(dd, dd);
    float b = dot(oc, dd);
    float c = dot(oc, oc) - Ra * Ra;
    float disc = b * b - a * c;
    if (disc <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    float sqd = sqrt(disc);
    float t0 = max((-b - sqd) / a, 0.0);
    float t1 = (-b + sqd) / a;
    float dp = b * b - a * (dot(oc, oc) - Rp * Rp);
    if (dp > 0.0) {
        float tp = (-b - sqrt(dp)) / a;
        if (tp > 0.0) t1 = min(t1, tp);
    }
    if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    const int NS = 8;
    float dt = (t1 - t0) / float(NS);
    float H = max(uRadius * uAtm.w * 0.28, 1e-6);
    float k = uAtmDen / H;
    float segLen = dt * sqrt(a);
    vec3 sum = vec3(0.0);
    float od = 0.0;
    float cosT = dot(rd, L);
    float phase = 0.375 * (1.0 + cosT * cosT) + 0.55 * pow(max(cosT, 0.0), 7.0);
    for (int i = 0; i < NS; i++) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 p = oc + dd * t;
        float hgt = max(length(p) - Rp, 0.0);
        float dens = exp(-hgt / H);
        float mu = dot(normalize(p), L);
        float vis = smoothstep(-0.18, 0.28, mu);
        vec3 sc = mix(vec3(1.0, 0.42, 0.18), vec3(1.0), smoothstep(-0.05, 0.45, mu));
        od += dens * segLen * k;
        sum += uAtm.rgb * sc * (dens * segLen * k * vis * exp(-od * 0.5));
    }
    return vec4(sum * phase * 1.05, exp(-od * 0.55));
}

void craters(vec3 p, out float depth, out float rim) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    float best = 1e9;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                vec3 g = vec3(float(x), float(y), float(z));
                vec3 id = ip + g;
                float h = hash13(id + vec3(17.17));
                if (h > 0.55) continue;
                vec3 o = hash33(id);
                float rad = mix(0.16, 0.46, (h / 0.55) * (h / 0.55));
                float q = length(g + o - fp) / rad;
                best = min(best, q);
            }
        }
    }
    depth = 1.0 - smoothstep(0.0, 0.92, best);
    depth *= depth;
    rim = smoothstep(0.78, 1.0, best) * (1.0 - smoothstep(1.0, 1.28, best));
}

vec3 bumpN(vec3 N, vec3 P, float h, float k) {
    vec3 dpdx = dFdx(P);
    vec3 dpdy = dFdy(P);
    float dhx = dFdx(h);
    float dhy = dFdy(h);
    vec3 r1 = cross(dpdy, N);
    vec3 r2 = cross(N, dpdx);
    float det = dot(dpdx, r1);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    return normalize(abs(det) * N - k * grad);
}

vec3 rotY(vec3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

void main() {
    vec3 N = normalize(vNrm);
    vec3 V = normalize(-vPos);
    vec3 d = normalize(vObj);
    vec3 L = normalize(uSun - vPos);

    // ---- resplandor aditivo (billboard) ----
    if (uType == 12) {
        float r = length(vObj.xz);
        if (r > 1.0) discard;
        float g1 = pow(1.0 - r, 2.6);
        float g2 = exp(-r * r * 40.0);
        float g3 = exp(-r * 6.0) * 0.35;
        float a = uPar.x * (g1 * 0.55 + g2 * 1.4 + g3);
        finalColor = vec4(uCol1 * a, 1.0);
        return;
    }
    // ---- cascaron de atmosfera (solo caras traseras, aditivo) ----
    if (uType == 11) {
        if (gl_FrontFacing) discard;
        vec3 Lc = normalize(uSun - uCenter);
        vec4 at = atmosphere(vec3(0.0), normalize(vPos), Lc);
        finalColor = vec4(at.rgb * uSunI, 1.0);
        return;
    }
    // ---- anillos ----
    if (uType == 10) {
        float r = length(vObj.xz);
)GLSL"
    R"GLSL(        if (r < uRingR.x || r > uRingR.y) discard;
        float dens = ringDensity(r);
        if (dens < 0.01) discard;
        float sh = shadowAt(vPos, L);
        float nl = dot(uRingN, L);
        float nv = dot(uRingN, V);
        float sameSide = (nl * nv > 0.0) ? 1.0 : 0.0;
        float bright = mix(0.30 * (1.0 - dens), 1.0, sameSide);
        vec3 base = mix(vec3(0.70, 0.60, 0.46), vec3(0.98, 0.91, 0.76), vnoise(vec3(r * 40.0, 3.0, 1.0)));
        if (uRingR.z > 0.5) base = vec3(0.30, 0.31, 0.33);
        vec3 col = base * (uSunI * pow(abs(nl), 0.6) * sh * bright + uAmb * 0.5);
        col = 1.0 - exp(-col * 1.35);
        finalColor = vec4(col, clamp(dens * 1.15, 0.0, 0.96) * smoothstep(0.0, 0.10, abs(nv)));
        return;
    }
    // ---- Sol ----
    if (uType == 5) {
        float t = uClock * 0.05;
        float g = fbm(d * 26.0 + vec3(0.0, t, 0.0));
        float sg = fbm(d * 7.0 + vec3(t * 0.3));
        float gran = smoothstep(0.28, 0.72, g * 0.7 + sg * 0.3);
        float spots = smoothstep(0.67, 0.75, fbm(d * 3.4 + vec3(9.0))) * (1.0 - smoothstep(0.35, 0.60, abs(d.y)));
        float mu = max(dot(N, V), 0.0);
        float limb = 1.0 - 0.55 * (1.0 - mu) - 0.15 * (1.0 - mu) * (1.0 - mu);
        vec3 col = mix(vec3(1.0, 0.60, 0.20), vec3(1.0, 0.95, 0.80), gran) * limb * 1.30;
        col = mix(col, col * vec3(0.50, 0.32, 0.18), spots * 0.85);
        col = mix(col * vec3(1.0, 0.66, 0.42), col, smoothstep(0.0, 0.55, mu));
        finalColor = vec4(min(col, vec3(1.7)), 1.0);
        return;
    }

    // ================= superficies iluminadas =================
    vec3 alb = vec3(0.5);
    vec3 emis = vec3(0.0);
    float hgt = 0.0;
    float specM = 0.0;
    float cloud = 0.0;
    float bumpK = uRadius * 0.012;

    if (uType == 0) {            // rocoso / helado con crateres
        vec3 p = d * uPar.x;
        float n = fbm(p * 1.3 + vec3(3.7));
        float n2 = fbm(p * 5.0 + vec3(11.0));
        alb = mix(uCol1, uCol2, smoothstep(0.36, 0.66, n));
        alb = mix(alb, uCol3, smoothstep(0.58, 0.85, n2) * 0.45);
        float cr1;
        float rm1;
        float cr2;
        float rm2;
        craters(p * 1.7, cr1, rm1);
        craters(p * 5.3 + vec3(4.0), cr2, rm2);
        float cr = cr1 + 0.6 * cr2;
        float rm = rm1 + 0.5 * rm2;
        alb *= 1.0 - uPar.y * 0.30 * cr;
        alb += vec3(uPar.y * 0.09 * rm);
        hgt = uPar.y * (rm * 0.55 - cr * 0.9) + 0.3 * n;
        float lat = abs(d.y);
        float ice = smoothstep(1.0 - uPar.z, 1.0 - uPar.z + 0.08, lat + (n2 - 0.5) * 0.12) * step(0.001, uPar.z);
        alb = mix(alb, vec3(0.92, 0.94, 0.96), ice);
        if (uPar.w > 0.5 && uPar.w < 1.5) {          // manchas volcanicas (Io)
            float sp = smoothstep(0.69, 0.74, vnoise(p * 2.4 + vec3(2.0)));
            alb = mix(alb, vec3(0.30, 0.15, 0.07), sp * 0.80);
            float pl = smoothstep(0.74, 0.80, vnoise(p * 3.5 + vec3(9.0)));
            alb = mix(alb, vec3(0.95, 0.94, 0.88), pl * 0.65);
            alb *= 0.88 + 0.24 * fbm(p * 2.0 + vec3(5.0));
        }
        if (uPar.w > 1.5) {                           // dos tonos (Japeto)
            float ft = smoothstep(-0.15, 0.15, d.x + 0.35 * (n - 0.5));
            alb = mix(uCol1, uCol2, ft) * (0.85 + 0.3 * n2);
        }
    } else if (uType == 1) {     // mundo con oceanos (Tierra)
        vec3 p = d * 2.1;
        float h = fbm(p + vec3(7.1)) * 0.78 + fbm(p * 3.3) * 0.22;
        float land = smoothstep(0.515, 0.535, h);
        float lat = abs(d.y);
        float pn = fbm(d * 6.0 + vec3(4.0));
        float thr = (d.y < 0.0) ? 0.88 : 0.945;
        float ice = smoothstep(thr, thr + 0.045, lat + (pn - 0.5) * 0.14);
        vec3 ocean = mix(vec3(0.010, 0.040, 0.150), vec3(0.035, 0.170, 0.400), smoothstep(0.38, 0.54, h));
        vec3 landc = mix(vec3(0.10, 0.27, 0.07), vec3(0.42, 0.36, 0.20), smoothstep(0.54, 0.72, h));
        float desert = (1.0 - smoothstep(0.0, 0.32, abs(lat - 0.30))) * smoothstep(0.42, 0.62, fbm(p * 1.8 + vec3(5.0)));
        landc = mix(landc, vec3(0.74, 0.65, 0.40), desert);
        landc = mix(landc, vec3(0.92, 0.93, 0.95), smoothstep(0.76, 0.88, h));
        alb = mix(ocean, landc, land);
        alb = mix(alb, vec3(0.93, 0.96, 1.0), ice);
        specM = (1.0 - land) * (1.0 - ice);
        hgt = land * (h - 0.55) * 2.0;
        vec3 cp = rotY(d, uTime * 0.12);
        float cl = fbm(cp * 3.2 + vec3(20.0)) + 0.18 * (fbm(cp * 9.0) - 0.5);
        cloud = smoothstep(0.575, 0.78, cl) * 0.88;
        float lights = land * (1.0 - ice) * smoothstep(0.58, 0.86, fbm(p * 14.0 + vec3(31.0)));
        emis = vec3(1.0, 0.72, 0.38) * lights * 0.9 * (1.0 - cloud * 0.8);
    } else if (uType == 3) {     // gigante gaseoso
        float lat = d.y;
        float turb = fbm(vec3(d.x * 4.0, lat * 12.0, d.z * 4.0) + vec3(2.0));
        float w = (turb - 0.5) * 0.9 * uPar.y;
        float zon = 0.5 + 0.5 * sin(lat * uPar.x * 2.3 + w * 5.0 + 5.0 * fbm(vec3(0.0, lat * 3.0, 1.0)));
        float bnd = fbm(vec3(d.x * 2.2, lat * uPar.x * 1.4 + w * 2.0, d.z * 2.2) + vec3(5.0));
        float fine = vnoise(vec3(d.x * 5.0, lat * uPar.x * 5.0 + w * 3.0, d.z * 5.0) + vec3(1.0));
        float v = mix(zon, clamp((bnd - 0.25) * 2.0, 0.0, 1.0), 0.40) + (fine - 0.5) * 0.18 * uPar.y;
        v = smoothstep(0.15, 0.85, clamp(v, 0.0, 1.0));
        alb = mix(uCol1, uCol2, v);
        alb = mix(alb, uCol3, smoothstep(0.62, 0.92, turb) * (0.25 + 0.5 * uPar.y));
        if (uPar.z > 0.0) {
            float lon = atan(d.z, d.x + 1e-6);
            float la = asin(clamp(d.y, -1.0, 1.0));
            float lat0 = -0.40;
            float dl = mod(lon - 0.6 + PI, 2.0 * PI) - PI;
            float dla = la - lat0;
            float rr = length(vec2(dl * cos(lat0) / 0.33, dla / 0.17));
            float sw = fbm(vec3(dl * 5.0, dla * 9.0, 3.0));
            float spot = (1.0 - smoothstep(0.35, 1.0, rr + (sw - 0.5) * 0.35));
)GLSL"
    R"GLSL(            vec3 spotCol = mix(vec3(0.72, 0.28, 0.16), vec3(0.04, 0.09, 0.30), uPar.w);
            alb = mix(alb, spotCol, spot * 0.9 * uPar.z);
        }
    } else if (uType == 4) {     // cubierta de nubes / neblina
        vec3 p = d * uPar.x * 1.5;
        float w = fbm(p * 1.2 + vec3(1.0));
        float n = fbm(p + 1.6 * vec3(w, w * 0.5, -w));
        float streak = fbm(vec3(d.x * 1.5, d.y * 6.0, d.z * 1.5) + vec3(w));
        alb = mix(uCol1, uCol2, smoothstep(0.30, 0.70, n));
        alb = mix(alb, uCol3, smoothstep(0.55, 0.85, streak) * 0.5);
    }

    // normal con relieve / cuerpos irregulares
    if (uLump > 0.0) {
        vec3 fn = normalize(cross(dFdx(vPos), dFdy(vPos)));
        if (dot(fn, N) < 0.0) fn = -fn;
        N = normalize(mix(N, fn, clamp(uLump, 0.0, 1.0)));
    }
    if (uType == 0 || uType == 1) N = bumpN(N, vPos, hgt, bumpK);

    float sh = shadowAt(vPos, L);
    if (uRingR.y > 0.0) {
        float dn = dot(uRingN, L);
        if (abs(dn) > 0.002) {
            float t = dot(uCenter - vPos, uRingN) / dn;
            if (t > 0.0) {
                vec3 hit = vPos + L * t;
                float rr = length(hit - uCenter) / uRadius;
                if (rr > uRingR.x && rr < uRingR.y) sh *= 1.0 - 0.88 * ringDensity(rr);
            }
        }
    }
    float ndl = dot(N, L);
    float nd = max(ndl, 0.0);
    float dif;
    if (uType == 0) {
        float nv = max(dot(N, V), 0.04);
        dif = mix(nd, 1.7 * nd / (nd + nv), 0.65);
    } else {
        dif = pow(nd, 0.85);
    }
    if (cloud > 0.0) {
        alb = mix(alb, vec3(0.96, 0.97, 1.0), cloud);
        specM *= (1.0 - cloud);
    }
    vec3 sunCol = vec3(1.0, 0.97, 0.92);
    vec3 col = alb * sunCol * (uSunI * dif * sh) + alb * uAmb + emis * (1.0 - smoothstep(-0.10, 0.12, ndl));
    if (specM > 0.0) {
        vec3 Hh = normalize(L + V);
        col += vec3(1.0, 0.95, 0.85) * (pow(max(dot(N, Hh), 0.0), 380.0) * specM * sh * uSunI * 0.55);
    }
    if (uAtm.w > 0.0) {
        vec3 Lc = normalize(uSun - uCenter);
        vec4 at = atmosphere(vec3(0.0), normalize(vPos), Lc);
        col = col * at.a + at.rgb * uSunI;
    }
    col = 1.0 - exp(-col * 1.35);
    finalColor = vec4(col, 1.0);
}
)GLSL";

static const std::string kGlslSkyFS =
    R"GLSL(out vec4 finalColor;
uniform vec2 uRes;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform float uTanHalf;
uniform float uPixAng;
uniform vec3 uGalN;
uniform vec3 uGalX;
uniform vec3 uGalY;

vec3 starLayer(vec3 d, float freq, float thr, float gain, float seed) {
    float pxr = uPixAng * freq * 0.9;
    if (pxr > 0.34) return vec3(0.0);
    vec3 q = d * freq;
    vec3 id = floor(q);
    vec3 f = q - id;
    vec3 h = hash33(id + vec3(seed));
    float m = hash13(id * 1.37 + vec3(seed + 5.1));
    vec3 sp = vec3(0.18) + 0.64 * h;
    float dist = length(f - sp);
    float rad = max(pxr, 0.012);
    float star = 1.0 - smoothstep(0.0, rad, dist);
    float vis = clamp((m - thr) / (1.0 - thr), 0.0, 1.0);
    float tcol = hash13(id + vec3(seed + 9.7));
    vec3 tint = mix(vec3(1.0, 0.72, 0.50), vec3(0.62, 0.78, 1.0), tcol);
    tint = mix(tint, vec3(1.0), 0.45);
    return tint * (star * vis * vis * gain);
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
    vec3 d = normalize(uFwd + uRight * (ndc.x * uTanHalf * uRes.x / uRes.y) + uUp * (ndc.y * uTanHalf));

    float b = dot(d, uGalN);
    float cx = dot(d, uGalX);
    float cy = dot(d, uGalY);
    float lon = atan(cy, cx);
    float band = exp(-b * b * 55.0);
    float core = exp(-lon * lon * 1.1) * exp(-b * b * 16.0);
    float n1 = fbm(d * 7.0);
    float n2 = fbm(d * 19.0 + vec3(3.0));
    float dust = smoothstep(0.50, 0.70, fbm(d * 11.0 + vec3(5.0))) * exp(-b * b * 90.0);
    float mw = band * (0.30 + 0.95 * n1) + core * 0.85;
    mw *= 1.0 - 0.72 * dust;
    vec3 mwCol = mix(vec3(0.42, 0.50, 0.78), vec3(1.0, 0.78, 0.52), clamp(core * 1.3, 0.0, 1.0));
    vec3 col = mwCol * mw * 0.22 * (0.6 + 0.8 * n2);

    float dens = 1.0 + 2.2 * band;
    col += starLayer(d, 38.0, 0.90, 2.2, 1.0);
    col += starLayer(d, 85.0, 0.93 - 0.02 * dens, 1.4, 2.0);
    col += starLayer(d, 190.0, 0.95 - 0.02 * dens, 1.0, 3.0);
    col += starLayer(d, 420.0, 0.965 - 0.02 * dens, 0.8, 4.0);
    float gr = hash13(floor(d * 700.0) + vec3(11.0));
    col += mwCol * band * smoothstep(0.80, 1.0, gr) * 0.10 * (0.4 + n1);
    col += vec3(0.0025, 0.0032, 0.0065);
    col = 1.0 - exp(-col * 1.7);
    finalColor = vec4(col, 1.0);
}
)GLSL";


#ifndef SOLAR_PHYSICS_ONLY
// ============================================================================
//  GRAFICOS 3D  (camara relativa en doble precision: el origen del render es la camara)
//  Unidad de render = 1000 km (Mm).  Los cuerpos se dibujan de lejos a cerca
//  (algoritmo del pintor) con planos near/far adaptativos.
// ============================================================================
struct ViewOpts {
    bool orbits = true, trails = true, labels = true, grid = false, axes = false, swarm = true, atmos = true, moonOrbits = true;
    float exposure = 1.0f, ambient = 0.025f;
    int selected = -1;
    double simDays = 0;
};
struct FrameCtx {
    V3 cam, right, up, fwd;
    double tanHalf = 0.4, focalPx = 1000, mm = AU_KM * 1e-3, nearP = 0.01, farP = 1e10;
    int W = 1280, H = 720;
};
struct Trail { std::vector<V3> p; int head = 0, n = 0; double last = -1e300, period = 0; };
static const int kTrailCap = 256;

struct BodyLocs { int type, c1, c2, c3, par, atm, atmDen, center, radius, sun, sunI, sunAng, amb, time, clock, lump, ringN, ringR, occ, occN, seed, flat; };
struct SkyLocs { int res, right, up, fwd, tanHalf, pixAng, galN, galX, galY; };
struct Gfx {
    Shader body{}, sky{};
    bool okBody = false, okSky = false;
    BodyLocs L{}; SkyLocs K{};
    Mesh sphHi{}, sphLo{}, sphLump{}, ringM{}, quadM{};
    Material mat{}, matDef{};
    Font font{};
    bool customFont = false;
    V3 galN{0, 0, 1}, galX{1, 0, 0}, galY{0, 1, 0};
    std::string status;
};

static bool shaderOk(const Shader& s) { return s.id > 0 && s.locs != nullptr; }
static Mesh makeRingMesh(float r0, float r1, int seg) {
    Mesh m = {};
    m.vertexCount = (seg + 1) * 2; m.triangleCount = seg * 2;
    m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
    m.normals = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
    m.texcoords = (float*)MemAlloc(m.vertexCount * 2 * sizeof(float));
    m.indices = (unsigned short*)MemAlloc(m.triangleCount * 3 * sizeof(unsigned short));
    for (int j = 0; j <= seg; j++) {
        float a = 2.0f * (float)kPI * j / seg, c = std::cos(a), s = std::sin(a);
        for (int k = 0; k < 2; k++) {
            int v = j * 2 + k; float r = k ? r1 : r0;
            m.vertices[v * 3] = r * c; m.vertices[v * 3 + 1] = 0; m.vertices[v * 3 + 2] = r * s;
            m.normals[v * 3] = 0; m.normals[v * 3 + 1] = 1; m.normals[v * 3 + 2] = 0;
            m.texcoords[v * 2] = (float)k; m.texcoords[v * 2 + 1] = (float)j / seg;
        }
    }
    for (int j = 0; j < seg; j++) {
        int a = j * 2; unsigned short* t = m.indices + j * 6;
        t[0] = (unsigned short)a; t[1] = (unsigned short)(a + 1); t[2] = (unsigned short)(a + 2);
        t[3] = (unsigned short)(a + 1); t[4] = (unsigned short)(a + 3); t[5] = (unsigned short)(a + 2);
    }
    UploadMesh(&m, false);
    return m;
}
static Matrix basisMatrix(V3 x, V3 y, V3 z, V3 t) {
    Matrix m = {};
    m.m0 = (float)x.x; m.m1 = (float)x.y; m.m2 = (float)x.z;
    m.m4 = (float)y.x; m.m5 = (float)y.y; m.m6 = (float)y.z;
    m.m8 = (float)z.x; m.m9 = (float)z.y; m.m10 = (float)z.z;
    m.m12 = (float)t.x; m.m13 = (float)t.y; m.m14 = (float)t.z; m.m15 = 1.0f;
    return m;
}
static void setF(Shader& s, int loc, float v) { SetShaderValue(s, loc, &v, SHADER_UNIFORM_FLOAT); }
static void setI(Shader& s, int loc, int v) { SetShaderValue(s, loc, &v, SHADER_UNIFORM_INT); }
static void set3(Shader& s, int loc, float a, float b, float c) { float v[3] = {a, b, c}; SetShaderValue(s, loc, v, SHADER_UNIFORM_VEC3); }
static void set4(Shader& s, int loc, float a, float b, float c, float d) { float v[4] = {a, b, c, d}; SetShaderValue(s, loc, v, SHADER_UNIFORM_VEC4); }
static void set3v(Shader& s, int loc, const V3& v) { set3(s, loc, (float)v.x, (float)v.y, (float)v.z); }

void gfxInit(Gfx& g) {
    std::string vs = "#version 330\n" + kGlslNoise + kGlslBodyVS, fs = "#version 330\n" + kGlslNoise + kGlslBodyFS;
    g.body = LoadShaderFromMemory(vs.c_str(), fs.c_str());
    g.okBody = shaderOk(g.body);
    std::string sf = "#version 330\n" + kGlslNoise + kGlslSkyFS;
    g.sky = LoadShaderFromMemory(nullptr, sf.c_str());   // vertice por defecto de raylib (todos los atributos validos)
    g.okSky = shaderOk(g.sky);
    if (g.okBody) {
        Shader& s = g.body; BodyLocs& L = g.L;
        L.type = GetShaderLocation(s, "uType"); L.c1 = GetShaderLocation(s, "uCol1"); L.c2 = GetShaderLocation(s, "uCol2"); L.c3 = GetShaderLocation(s, "uCol3");
        L.par = GetShaderLocation(s, "uPar"); L.atm = GetShaderLocation(s, "uAtm"); L.atmDen = GetShaderLocation(s, "uAtmDen");
        L.center = GetShaderLocation(s, "uCenter"); L.radius = GetShaderLocation(s, "uRadius"); L.sun = GetShaderLocation(s, "uSun");
        L.sunI = GetShaderLocation(s, "uSunI"); L.sunAng = GetShaderLocation(s, "uSunAng"); L.amb = GetShaderLocation(s, "uAmb");
        L.time = GetShaderLocation(s, "uTime"); L.clock = GetShaderLocation(s, "uClock"); L.lump = GetShaderLocation(s, "uLump");
        L.ringN = GetShaderLocation(s, "uRingN"); L.ringR = GetShaderLocation(s, "uRingR"); L.occ = GetShaderLocation(s, "uOcc");
        L.occN = GetShaderLocation(s, "uOccN"); L.seed = GetShaderLocation(s, "uSeed"); L.flat = GetShaderLocation(s, "uFlat");
        g.mat = LoadMaterialDefault(); g.mat.shader = g.body;
    } else g.status = "Shaders no disponibles (se necesita OpenGL 3.3): modo basico";
    if (g.okSky) {
        Shader& s = g.sky; SkyLocs& K = g.K;
        K.res = GetShaderLocation(s, "uRes"); K.right = GetShaderLocation(s, "uRight"); K.up = GetShaderLocation(s, "uUp"); K.fwd = GetShaderLocation(s, "uFwd");
        K.tanHalf = GetShaderLocation(s, "uTanHalf"); K.pixAng = GetShaderLocation(s, "uPixAng");
        K.galN = GetShaderLocation(s, "uGalN"); K.galX = GetShaderLocation(s, "uGalX"); K.galY = GetShaderLocation(s, "uGalY");
    }
    g.matDef = LoadMaterialDefault();
    g.sphHi = GenMeshSphere(1.0f, 96, 128); g.sphLo = GenMeshSphere(1.0f, 28, 40); g.sphLump = GenMeshSphere(1.0f, 64, 96);
    g.ringM = makeRingMesh(1.0f, 2.6f, 256); g.quadM = GenMeshPlane(2.0f, 2.0f, 1, 1);
    // base galactica (polo norte y centro galactico, J2000) en coordenadas eclipticas
    g.galN = poleFromRaDec(192.85948, 27.12825); g.galX = poleFromRaDec(266.40499, -28.93617);
    g.galX = vnorm(g.galX - g.galN * dot(g.galX, g.galN)); g.galY = cross(g.galN, g.galX);
    // fuente con tildes: se intenta una TTF del sistema; si no, la fuente por defecto de raylib (Latin-1)
    static const char* paths[] = {"C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"};
    g.font = GetFontDefault();
    int cps[224]; int nc = 0;
    for (int c = 32; c < 127; c++) cps[nc++] = c;
    for (int c = 160; c < 256; c++) cps[nc++] = c;
    for (const char* p : paths) if (FileExists(p)) {
        Font f = LoadFontEx(p, 34, cps, nc);
        if (f.texture.id > 0) { g.font = f; g.customFont = true; SetTextureFilter(f.texture, TEXTURE_FILTER_BILINEAR); break; }
    }
}

// ---------------- utilidades de proyeccion ----------------
static inline V3 relMm(const FrameCtx& f, const V3& wAU) { return (wAU - f.cam) * f.mm; }
static inline bool projectRel(const FrameCtx& f, const V3& rel, Vector2& s, double& z) {
    z = dot(rel, f.fwd);
    if (z <= 1e-12) return false;
    double x = dot(rel, f.right) / z, y = dot(rel, f.up) / z;
    s.x = (float)(f.W * 0.5 + x * f.focalPx); s.y = (float)(f.H * 0.5 - y * f.focalPx);
    return true;
}
static inline void lv(const FrameCtx& f, const V3& wAU, Color c) {
    rlColor4ub(c.r, c.g, c.b, c.a);
    V3 r = relMm(f, wAU);
    rlVertex3f((float)r.x, (float)r.y, (float)r.z);
}
static inline unsigned char clampA(double a) { return (unsigned char)(a < 0 ? 0 : (a > 255 ? 255 : a)); }

// ---------------- orientacion de un cuerpo ----------------
void bodyBasis(Sim& S, int i, V3& X, V3& Y, V3& Z) {
    const Body& b = S.b[i];
    if (b.kind == K_MOON && b.tidalLock && b.moonSlot >= 0) {
        const Prim& pr = S.p[b.prim];
        V3 r = pr.r[b.moonSlot], w = pr.w[b.moonSlot];
        V3 n = vnorm(cross(r, w)), toP = r * -1.0;
        V3 x = toP - n * dot(toP, n);
        Y = n; X = vnorm(x); Z = cross(X, Y);
        return;
    }
    Mat3 F = eqFrame(b.spin);
    double ph = b.dispPhase;
    X = F.c0 * std::cos(ph) + F.c1 * std::sin(ph); Y = F.c2; Z = cross(X, Y);
}

void drawSky(Gfx& g, const FrameCtx& f) {
    if (!g.okSky) return;
    Shader& s = g.sky;
    float res[2] = {(float)GetRenderWidth(), (float)GetRenderHeight()};
    SetShaderValue(s, g.K.res, res, SHADER_UNIFORM_VEC2);
    set3v(s, g.K.right, f.right); set3v(s, g.K.up, f.up); set3v(s, g.K.fwd, f.fwd);
    setF(s, g.K.tanHalf, (float)f.tanHalf); setF(s, g.K.pixAng, (float)(2.0 * f.tanHalf / std::max(1, GetRenderHeight())));
    set3v(s, g.K.galN, g.galN); set3v(s, g.K.galX, g.galX); set3v(s, g.K.galY, g.galY);
    BeginShaderMode(s);
    DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), WHITE);
    EndShaderMode();
}

// ---------------- cuerpos ----------------
static void setLook(Gfx& g, const Look& lk, int type) {
    BodyLocs& L = g.L; Shader& s = g.body;
    setI(s, L.type, type);
    set3(s, L.c1, lk.c1[0], lk.c1[1], lk.c1[2]); set3(s, L.c2, lk.c2[0], lk.c2[1], lk.c2[2]); set3(s, L.c3, lk.c3[0], lk.c3[1], lk.c3[2]);
    set4(s, L.par, lk.par[0], lk.par[1], lk.par[2], lk.par[3]);
    set4(s, L.atm, lk.atm[0], lk.atm[1], lk.atm[2], lk.atm[3]); setF(s, L.atmDen, lk.atm[4]);
    setF(s, L.lump, lk.lump);
}

void drawOneBody(Gfx& g, const FrameCtx& f, Sim& S, int i, const ViewOpts& vo, double distMm, double pxR) {
    Body& b = S.b[i];
    const double Rm = b.radiusKm * 1e-3;
    V3 c = relMm(f, b.pos);
    V3 X, Y, Z; bodyBasis(S, i, X, Y, Z);
    double flat = (b.kind == K_STAR || b.kind == K_PLANET) ? b.flat : 0.0;
    V3 sc = {Rm * b.axes.x, Rm * b.axes.z * (1.0 - flat), Rm * b.axes.y};
    if (b.kind == K_PLANET || b.kind == K_STAR) sc = {Rm, Rm * (1.0 - flat), Rm};
    Matrix model = basisMatrix(X * sc.x, Y * sc.y, Z * sc.z, c);
    Mesh& mesh = (b.look.lump > 0 ? g.sphLump : (pxR > 28 ? g.sphHi : g.sphLo));
    if (!g.okBody) {
        g.matDef.maps[MATERIAL_MAP_DIFFUSE].color = b.col;
        DrawMesh(g.sphLo, g.matDef, model);
        return;
    }
    Shader& s = g.body; BodyLocs& L = g.L;
    const Look& lk = b.look;
    V3 sunRel = relMm(f, S.b[0].pos);
    double dSunAU = std::max(vlen(b.pos - S.b[0].pos), 0.02);
    set3v(s, L.center, c); setF(s, L.radius, (float)Rm); set3v(s, L.sun, sunRel);
    setF(s, L.sunI, (float)(clampd(std::pow(dSunAU, -0.55), 0.55, 1.6) * vo.exposure));
    setF(s, L.sunAng, (float)(S.b[0].radiusKm / (dSunAU * AU_KM)));
    setF(s, L.amb, vo.ambient); setF(s, L.time, (float)vo.simDays); setF(s, L.clock, (float)GetTime());
    set3v(s, L.ringN, Y); set3(s, L.ringR, lk.ringIn, lk.ringOut, (float)lk.ringType);
    setF(s, L.seed, (float)(hashStr(b.key.c_str()) % 97)); setF(s, L.flat, (float)(1.0 / (1.0 - flat) - 1.0));
    // oclusores para eclipses: planeta padre y hermanas, o lunas del planeta
    float occ[24] = {0}; int no = 0;
    auto addOcc = [&](int j) {
        if (j == i || no >= 6 || !S.b[j].alive) return;
        V3 o = relMm(f, S.b[j].pos);
        occ[no * 4] = (float)o.x; occ[no * 4 + 1] = (float)o.y; occ[no * 4 + 2] = (float)o.z; occ[no * 4 + 3] = (float)(S.b[j].radiusKm * 1e-3 * 0.999);
        no++;
    };
    if (b.kind == K_MOON) { addOcc(b.parent); for (int mj : S.p[b.prim].moons) addOcc(mj); }
    else if (b.kind == K_PLANET && b.prim >= 0) for (int mj : S.p[b.prim].moons) addOcc(mj);
    SetShaderValueV(s, L.occ, occ, SHADER_UNIFORM_VEC4, 6); setI(s, L.occN, no);

    if (lk.type == 5) {   // Sol: esfera + resplandor
        setLook(g, lk, 5); DrawMesh(g.sphHi, g.mat, model);
        double glowR = std::max(Rm * 4.6, 22.0 / f.focalPx * distMm);
        Look gl; gl.c1[0] = 1.0f; gl.c1[1] = 0.80f; gl.c1[2] = 0.52f; gl.par[0] = 1.25f;
        setLook(g, gl, 12);
        V3 rr = f.right * glowR, uu = f.up * glowR;
        Matrix gm = basisMatrix(rr, f.fwd, uu, c);
        rlDisableBackfaceCulling(); rlDisableDepthMask(); BeginBlendMode(BLEND_ADDITIVE);
        DrawMesh(g.quadM, g.mat, gm);
        EndBlendMode(); rlEnableDepthMask(); rlEnableBackfaceCulling();
        return;
    }
    setLook(g, lk, lk.type);
    DrawMesh(mesh, g.mat, model);
    if (vo.atmos && lk.atm[3] > 0 && pxR > 3) {
        setI(s, L.type, 11);
        double k = 1.0 + lk.atm[3];
        Matrix sm = basisMatrix(X * (Rm * k), Y * (Rm * (1.0 - flat) * k), Z * (Rm * k), c);
        rlDisableBackfaceCulling(); rlDisableDepthMask(); BeginBlendMode(BLEND_ADDITIVE);
        DrawMesh(g.sphHi, g.mat, sm);
        EndBlendMode(); rlEnableDepthMask(); rlEnableBackfaceCulling();
    }
    if (lk.ringOut > 0 && pxR > 2) {
        setI(s, L.type, 10);
        Matrix rm = basisMatrix(X * Rm, Y * Rm, Z * Rm, c);
        rlDisableBackfaceCulling(); BeginBlendMode(BLEND_ALPHA);
        DrawMesh(g.ringM, g.mat, rm);
        EndBlendMode(); rlEnableBackfaceCulling();
    }
}

// ---------------- orbitas, estelas, cuadricula ----------------
static void drawOrbitLine(const FrameCtx& f, const V3& center, const Elems& el, Color col, double alphaBase, double alphaGain, int seg) {
    double a = el.a, e = el.e, b = a * std::sqrt(std::max(1.0 - e * e, 1e-9));
    double E0 = 2.0 * std::atan2(std::sqrt(1 - e) * std::sin(el.nu / 2), std::sqrt(1 + e) * std::cos(el.nu / 2));
    rlBegin(RL_LINES);
    V3 prev; Color pc{0, 0, 0, 0};
    for (int k = 0; k <= seg; k++) {
        double E = 2 * kPI * k / seg;
        V3 p = center + el.P * (a * (std::cos(E) - e)) + el.Q * (b * std::sin(E));
        double ph = std::fmod(E0 - E + 4 * kPI, 2 * kPI) / (2 * kPI), fa = (1.0 - ph) * (1.0 - ph);
        Color c = {col.r, col.g, col.b, clampA(alphaBase + alphaGain * fa)};
        if (k > 0) { lv(f, prev, pc); lv(f, p, c); }
        prev = p; pc = c;
    }
    rlEnd();
}
void drawOrbits(const FrameCtx& f, Sim& S, const ViewOpts& vo) {
    for (size_t i = 1; i < S.b.size(); i++) {
        Body& b = S.b[i];
        if (!b.alive || b.kind == K_STAR) continue;
        bool sel = ((int)i == vo.selected);
        if (b.kind == K_MOON && !(vo.moonOrbits || sel)) continue;
        V3 r, v, cen; double mu; Mat3 F;
        if (!S.relState((int)i, r, v, mu, cen, F)) continue;
        Elems el = elementsFromState(r, v, mu);
        if (!el.bound || el.a <= 0) continue;
        double dC = vlen(cen - f.cam) * f.mm + 1e-9, sizePx = (el.a * (1 + el.e) * f.mm) / dC * f.focalPx;
        if (!sel && (sizePx < 7 || sizePx > 60000)) continue;
        if (b.kind == K_MOON && !sel) {   // las orbitas de lunas solo cuando el sistema se ve de cerca
            const Body& pl = S.b[b.parent]; double dP = vlen(pl.pos - f.cam) * f.mm + 1e-9;
            if ((pl.radiusKm * 1e-3) / dP * f.focalPx < 1.2 && sizePx < 22) continue;
        }
        int seg = (int)clampd(sizePx * 0.5, 64, 360);
        double ab = (b.kind == K_MOON ? 55 : 70) * (sel ? 1.8 : 1.0), ag = (b.kind == K_MOON ? 110 : 150) * (sel ? 1.5 : 1.0);
        Color col = sel ? Color{255, 255, 255, 255} : b.col;
        drawOrbitLine(f, cen, el, col, ab, ag, seg);
    }
}
void drawTrails(const FrameCtx& f, Sim& S, const std::vector<Trail>& T, const ViewOpts& vo) {
    for (size_t i = 1; i < S.b.size() && i < T.size(); i++) {
        const Body& b = S.b[i];
        const Trail& t = T[i];
        if (!b.alive || t.n < 2) continue;
        V3 base = (b.kind == K_MOON) ? S.b[b.parent].pos : V3{};
        double gain = ((int)i == vo.selected) ? 1.0 : 0.65;
        rlBegin(RL_LINES);
        for (int k = 0; k + 1 < t.n; k++) {
            int ia = (t.head - t.n + k + kTrailCap * 2) % kTrailCap, ib = (ia + 1) % kTrailCap;
            double fa = (double)(k + 1) / t.n;
            Color ca = {b.col.r, b.col.g, b.col.b, clampA(230 * gain * fa * fa)};
            Color cb = {b.col.r, b.col.g, b.col.b, clampA(230 * gain * (fa + 1.0 / t.n) * (fa + 1.0 / t.n))};
            lv(f, base + t.p[ia], ca); lv(f, base + t.p[ib], cb);
        }
        // ultimo tramo hasta la posicion actual
        int il = (t.head - 1 + kTrailCap) % kTrailCap;
        lv(f, base + t.p[il], Color{b.col.r, b.col.g, b.col.b, clampA(230 * gain)}); lv(f, b.pos, Color{b.col.r, b.col.g, b.col.b, clampA(230 * gain)});
        rlEnd();
    }
}
void drawGrid(const FrameCtx& f, const V3& sunPos) {
    static const double R[] = {0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200, 500};
    double dS = vlen(sunPos - f.cam) * f.mm + 1e-9;
    for (double r : R) {
        double px = (r * f.mm) / dS * f.focalPx;
        if (px < 18 || px > 40000) continue;
        double alpha = clampd(70.0 * std::min(1.0, px / 120.0), 18, 70);
        Color c = {110, 150, 220, clampA(alpha)};
        int seg = (int)clampd(px, 90, 400);
        rlBegin(RL_LINES);
        for (int k = 0; k < seg; k++) {
            double a0 = 2 * kPI * k / seg, a1 = 2 * kPI * (k + 1) / seg;
            lv(f, sunPos + V3{std::cos(a0) * r, std::sin(a0) * r, 0}, c); lv(f, sunPos + V3{std::cos(a1) * r, std::sin(a1) * r, 0}, c);
        }
        rlEnd();
    }
    // ejes: direccion del equinoccio de marzo (+X) y polo eclíptico
    double L = std::max(dS / f.mm * 0.03, 0.01);
    rlBegin(RL_LINES);
    lv(f, sunPos, Color{220, 90, 90, 90}); lv(f, sunPos + V3{L, 0, 0}, Color{220, 90, 90, 90});
    lv(f, sunPos, Color{90, 220, 120, 90}); lv(f, sunPos + V3{0, L, 0}, Color{90, 220, 120, 90});
    lv(f, sunPos, Color{100, 140, 255, 90}); lv(f, sunPos + V3{0, 0, L}, Color{100, 140, 255, 90});
    rlEnd();
}
void drawAxesOf(const FrameCtx& f, Sim& S, int i) {
    const Body& b = S.b[i];
    V3 X, Y, Z; bodyBasis(S, i, X, Y, Z);
    double R = b.radiusKm / AU_KM;
    rlBegin(RL_LINES);
    lv(f, b.pos - Y * (R * 1.9), Color{90, 200, 255, 60}); lv(f, b.pos + Y * (R * 2.2), Color{90, 200, 255, 230});
    int seg = 128;
    for (int k = 0; k < seg; k++) {
        double a0 = 2 * kPI * k / seg, a1 = 2 * kPI * (k + 1) / seg;
        Color c = {255, 220, 120, 160};
        lv(f, b.pos + (X * std::cos(a0) + Z * std::sin(a0)) * (R * 1.5), c); lv(f, b.pos + (X * std::cos(a1) + Z * std::sin(a1)) * (R * 1.5), c);
    }
    lv(f, b.pos, Color{255, 120, 120, 200}); lv(f, b.pos + X * (R * 1.7), Color{255, 120, 120, 200});   // meridiano de referencia
    rlEnd();
}

// ---------------- enjambre (asteroides / Kuiper) ----------------
void drawSwarm(const FrameCtx& f, Sim& S) {
    static const Color tc[5] = {{210, 195, 175, 190}, {235, 160, 120, 200}, {205, 175, 150, 190}, {185, 140, 120, 180}, {120, 215, 255, 230}};
    rlDisableDepthMask(); BeginBlendMode(BLEND_ADDITIVE);
    rlBegin(RL_QUADS);
    for (const Particle& q : S.sw) {
        if (!q.alive) continue;
        V3 r = relMm(f, q.x);
        double z = dot(r, f.fwd);
        if (z < f.nearP * 1.01) continue;
        double px = clampd(q.size * 0.62, 1.0, 4.2), h = 0.5 * px * z / f.focalPx;
        Color c = tc[q.type > 4 ? 0 : q.type];
        double sh = 0.45 + 0.55 * q.shade / 255.0;
        rlColor4ub((unsigned char)(c.r * sh), (unsigned char)(c.g * sh), (unsigned char)(c.b * sh), c.a);
        V3 a = f.right * h, bb = f.up * h;
        V3 p0 = r - a - bb, p1 = r + a - bb, p2 = r + a + bb, p3 = r - a + bb;
        rlVertex3f((float)p0.x, (float)p0.y, (float)p0.z); rlVertex3f((float)p1.x, (float)p1.y, (float)p1.z);
        rlVertex3f((float)p2.x, (float)p2.y, (float)p2.z); rlVertex3f((float)p3.x, (float)p3.y, (float)p3.z);
    }
    rlEnd();
    EndBlendMode(); rlEnableDepthMask();
}

// ---------------- colas de cometas ----------------
void drawCometTail(const FrameCtx& f, Sim& S, int i) {
    const Body& b = S.b[i];
    V3 sunPos = S.b[0].pos, rel = b.pos - sunPos;
    double r = vlen(rel);
    if (r > 6.0) return;
    V3 away = vnorm(rel), vel = vnorm(b.vel - S.b[0].vel);
    double act = clampd(1.6 / (r * r), 0.0, 6.0), L = clampd(0.25 * std::pow(1.0 / std::max(r, 0.15), 1.5), 0.01, 1.6);
    V3 view = vnorm(b.pos - f.cam);
    auto tail = [&](V3 dir, V3 bend, double len, double w0, Color col, int seg, double amp) {
        V3 side = vnorm(cross(dir, view));
        rlBegin(RL_QUADS);
        for (int k = 0; k < seg; k++) {
            double t0 = (double)k / seg, t1 = (double)(k + 1) / seg;
            auto P = [&](double t) { return b.pos + dir * (len * t) + bend * (len * t * t * 0.55); };
            auto W = [&](double t) { return w0 * (1.0 - 0.85 * t) + len * 0.004; };
            double a0 = amp * (1.0 - t0) * (1.0 - t0), a1 = amp * (1.0 - t1) * (1.0 - t1);
            V3 p0 = P(t0), p1 = P(t1);
            Color c0 = {col.r, col.g, col.b, clampA(a0)}, c1 = {col.r, col.g, col.b, clampA(a1)};
            lv(f, p0 - side * W(t0), c0); lv(f, p0 + side * W(t0), c0); lv(f, p1 + side * W(t1), c1); lv(f, p1 - side * W(t1), c1);
        }
        rlEnd();
    };
    rlDisableBackfaceCulling(); rlDisableDepthMask(); BeginBlendMode(BLEND_ADDITIVE);
    double w = L * 0.03 + 4e-5;
    tail(away, V3{}, L, w, Color{110, 170, 255, 255}, 28, 150.0 * clampd(act, 0.1, 1.0));                         // cola ionica (recta)
    V3 bend = vel * -1.0;
    tail(vnorm(away + bend * 0.12), bend * 0.55, L * 0.8, w * 2.6, Color{255, 232, 190, 255}, 28, 120.0 * clampd(act, 0.1, 1.0)); // cola de polvo (curva)
    EndBlendMode(); rlEnableDepthMask(); rlEnableBackfaceCulling();
}

// ---------------- pasada 3D completa ----------------
struct DrawItem { double d; int i; };
void render3D(Gfx& g, FrameCtx& f, Sim& S, const std::vector<Trail>& trails, const ViewOpts& vo, std::vector<double>& pxOut) {
    // planos near/far adaptativos
    double nearest = 1e30;
    pxOut.assign(S.b.size(), 0.0);
    std::vector<DrawItem> items;
    for (size_t i = 0; i < S.b.size(); i++) {
        const Body& b = S.b[i];
        if (!b.alive) continue;
        double Rm = b.radiusKm * 1e-3 * std::max(b.axes.x, 1.0), d = vlen(relMm(f, b.pos));
        double pxR = Rm / std::max(d, 1e-9) * f.focalPx;
        pxOut[i] = pxR;
        nearest = std::min(nearest, std::max(d - Rm * (1.0 + (b.look.atm[3] > 0 ? b.look.atm[3] : 0.0)), 0.0005));
        items.push_back({d, (int)i});
    }
    f.nearP = clampd(nearest * 0.4, 1e-4, 1e8); f.farP = 1e10;
    rlSetClipPlanes(f.nearP, f.farP);
    Camera3D c3 = {};
    c3.position = Vector3{0, 0, 0}; c3.target = Vector3{(float)f.fwd.x, (float)f.fwd.y, (float)f.fwd.z};
    c3.up = Vector3{(float)f.up.x, (float)f.up.y, (float)f.up.z}; c3.fovy = (float)(2.0 * std::atan(f.tanHalf) / kDEG); c3.projection = CAMERA_PERSPECTIVE;
    BeginMode3D(c3);
    if (vo.grid) drawGrid(f, S.b[0].pos);
    if (vo.orbits) drawOrbits(f, S, vo);
    if (vo.trails) drawTrails(f, S, trails, vo);
    if (vo.axes && vo.selected > 0 && vo.selected < (int)S.b.size() && S.b[vo.selected].alive) drawAxesOf(f, S, vo.selected);
    if (vo.swarm) drawSwarm(f, S);
    std::sort(items.begin(), items.end(), [](const DrawItem& a, const DrawItem& b) { return a.d > b.d; });
    for (const DrawItem& it : items) {
        const Body& b = S.b[it.i];
        double pxR = pxOut[it.i];
        bool isSun = (b.kind == K_STAR);
        if (b.kind == K_COMET) drawCometTail(f, S, it.i);
        if (pxR >= 0.7 || isSun) drawOneBody(g, f, S, it.i, vo, it.d, pxR);
    }
    EndMode3D();
}
#endif

#ifndef SOLAR_PHYSICS_ONLY
// ============================================================================
//  APLICACION: camara orbital (raton / tactil / teclado), interfaz, edicion y bucle principal
// ============================================================================
static const double kSpeeds[] = {1, 10, 60, 600, 3600, 21600, 86400, 259200, 864000, 2592000, 7776000, 31557600, 94672800, 315576000, 946728000, 3155760000.0};
static const int kNumSpeeds = (int)(sizeof(kSpeeds) / sizeof(kSpeeds[0]));
static const char* kMeses[12] = {"ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"};

static const char* fmt(const char* f, ...) {
    static char buf[8][256]; static int k = 0;
    char* b = buf[k = (k + 1) & 7];
    va_list ap; va_start(ap, f); std::vsnprintf(b, 256, f, ap); va_end(ap);
    return b;
}
static std::string fmtSpeed(double s) {
    if (s >= 31557600.0 * 0.999) return fmt("%.3g años/s", s / 31557600.0);
    if (s >= 86400.0 * 0.999) return fmt("%.3g días/s", s / 86400.0);
    if (s >= 3600.0 * 0.999) return fmt("%.3g h/s", s / 3600.0);
    if (s >= 60.0 * 0.999) return fmt("%.3g min/s", s / 60.0);
    return fmt("%.3g x (tiempo real)", s);
}

struct UIState {
    int id = 0, active = -1, pressed = -1;
    float s = 1.0f;
    Vector2 m{0, 0};
    bool down = false, pressedNow = false, released = false;
};
struct App {
    Sim S; Gfx g; ViewOpts vo; BuildOpts bo;
    std::vector<Trail> trails; std::vector<int> lookSel;
    std::vector<Vector2> scr; std::vector<double> pxr;
    // tiempo
    bool playing = true; int speedIdx = 6, dir = 1; double achieved = 0, jdStart = 0; bool limited = false, degraded = false;
    // camara
    double yaw = 0.7, pitch = 0.5, logDist = std::log(6.0), fromLog = 0, toLog = 0, tt = 0;
    int focus = 0; V3 pan, fpos, fromPos; bool trans = false;
    FrameCtx fc;
    // interfaz
    float uiUser = 1.0f; bool showUI = true, showList = true, showHelp = true, panMode = false;
    float scrollR = 0, scrollL = 0; std::string toast; double toastT = 0;
    bool dragScene = false; Vector2 pressPos{0, 0}; double lastClickT = 0; int lastClickBody = -1;
    double tAcc = 0, qualityT = 0; float fps = 60; int swarmLevel = 2;
    Rng rng{99};
};
static App A;
static UIState U;
static Rectangle gClip = {0, 0, 100000, 100000};
static double gmEarth() { return gmConv(398600.436); }
static float Sc(float x) { return x * U.s; }

static void say(const std::string& s) { A.toast = s; A.toastT = GetTime(); }
static double nowJD() { return (double)std::time(nullptr) / 86400.0 + 2440587.5; }
static int findKey(const char* k) { for (size_t i = 0; i < A.S.b.size(); i++) if (A.S.b[i].key == k && A.S.b[i].alive) return (int)i; return -1; }

static void ensureTrails() {
    while (A.trails.size() < A.S.b.size()) { Trail t; t.p.assign(kTrailCap, V3{}); A.trails.push_back(t); }
    if (A.lookSel.size() < A.S.b.size()) A.lookSel.resize(A.S.b.size(), -1);
}
static void clearTrail(int i) { if (i >= 0 && i < (int)A.trails.size()) { A.trails[i].n = 0; A.trails[i].head = 0; A.trails[i].last = -1e300; } }
static void clearTrails() { for (size_t i = 0; i < A.trails.size(); i++) clearTrail((int)i); }
static void updatePeriods() {
    ensureTrails();
    for (size_t i = 1; i < A.S.b.size(); i++) {
        A.trails[i].period = 0;
        if (!A.S.b[i].alive) continue;
        V3 r, v, c; double mu; Mat3 F;
        if (!A.S.relState((int)i, r, v, mu, c, F)) continue;
        Elems el = elementsFromState(r, v, mu);
        if (el.bound) A.trails[i].period = el.T;
    }
}
static void sampleTrails() {
    Sim& S = A.S;
    for (size_t i = 1; i < S.b.size() && i < A.trails.size(); i++) {
        Trail& t = A.trails[i]; const Body& b = S.b[i];
        if (!b.alive || t.period < 8.0 || t.period > 2e6) continue;
        double dtS = t.period / 200.0;
        if (std::fabs(S.t - t.last) < dtS) continue;
        V3 p = (b.kind == K_MOON) ? b.pos - S.b[b.parent].pos : b.pos;
        t.p[t.head] = p; t.head = (t.head + 1) % kTrailCap; if (t.n < kTrailCap) t.n++; t.last = S.t;
    }
}
static void focusOn(int i, bool autoZoom) {
    if (i < 0 || i >= (int)A.S.b.size()) return;
    A.fromPos = A.fpos; A.fromLog = A.logDist; A.focus = i; A.pan = V3{};
    double R = A.S.b[i].radiusKm / AU_KM;
    A.toLog = autoZoom ? std::log(std::max(R * (A.S.b[i].kind == K_STAR ? 9.0 : 5.5), 1e-9)) : A.logDist;
    if (A.S.b[i].kind == K_COMET || A.S.b[i].kind == K_SMALL) A.toLog = std::log(std::max(R * 9.0, 2e-8));
    A.trans = true; A.tt = 0;
}
static void rebuild(double jd) {
    buildSolarSystem(A.S, jd, A.bo);
    A.jdStart = jd; A.trails.clear(); A.lookSel.clear(); ensureTrails(); updatePeriods();
    A.S.stepHook = []() { sampleTrails(); };
    A.vo.selected = -1; A.focus = 0; A.pan = V3{}; A.fpos = A.S.b[0].pos; A.trans = false; A.logDist = std::log(7.0);
    A.scr.assign(A.S.b.size(), Vector2{0, 0}); A.pxr.assign(A.S.b.size(), 0.0);
}
static void applySwarmLevel() {
    static const int lv[5][6] = {{0, 0, 0, 0, 0, 0}, {1200, 300, 80, 600, 40, 10}, {6000, 1400, 300, 2800, 150, 40}, {14000, 2800, 600, 6000, 300, 80}, {28000, 5000, 1200, 12000, 600, 150}};
    const int* l = lv[A.swarmLevel];
    A.bo.nMain = l[0]; A.bo.nTroj = l[1]; A.bo.nHilda = l[2]; A.bo.nKuiper = l[3]; A.bo.nNeo = l[4]; A.bo.nCentaur = l[5];
}

// ---------------- primitivas de interfaz ----------------
static void txt(const char* s, float x, float y, float size, Color c) {
    DrawTextEx(A.g.font, s, Vector2{x, y}, size, A.g.customFont ? 0.5f : 1.0f, c);
}
static float txtW(const char* s, float size) { return MeasureTextEx(A.g.font, s, size, A.g.customFont ? 0.5f : 1.0f).x; }
static bool inRect(Vector2 p, Rectangle r) { return CheckCollisionPointRec(p, r); }
static void panelBox(Rectangle r) {
    DrawRectangleRounded(r, 0.025f, 8, Color{10, 14, 24, 215});
    DrawRectangleLinesEx(r, 1.0f, Color{110, 140, 200, 70});
}
static bool uiButton(Rectangle r, const char* label, bool active = false, bool enabled = true) {
    int id = ++U.id;
    bool over = enabled && inRect(U.m, r) && inRect(U.m, gClip);
    if (over && U.pressedNow) U.pressed = id;
    bool click = over && U.released && U.pressed == id;
    Color bg = !enabled ? Color{26, 28, 36, 170} : active ? Color{60, 112, 200, 240} : over ? Color{58, 70, 98, 240} : Color{32, 40, 60, 230};
    DrawRectangleRounded(r, 0.28f, 6, bg);
    DrawRectangleLinesEx(r, 1.0f, Color{130, 155, 205, active ? (unsigned char)200 : (unsigned char)90});
    float fs = Sc(15); Vector2 ts = MeasureTextEx(A.g.font, label, fs, A.g.customFont ? 0.5f : 1.0f);
    txt(label, r.x + (r.width - ts.x) * 0.5f, r.y + (r.height - ts.y) * 0.5f, fs, enabled ? Color{235, 240, 250, 255} : Color{120, 125, 140, 255});
    return click;
}
static bool uiSlider(Rectangle r, const char* label, const char* valText, double& v, double lo, double hi, bool logS) {
    int id = ++U.id;
    txt(label, r.x, r.y, Sc(13), Color{185, 198, 225, 255});
    float vw = txtW(valText, Sc(13));
    txt(valText, r.x + r.width - vw, r.y, Sc(13), Color{255, 255, 255, 255});
    Rectangle tr = {r.x, r.y + Sc(23), r.width, Sc(6)}, hit = {r.x - Sc(4), r.y + Sc(15), r.width + Sc(8), Sc(24)};
    bool over = inRect(U.m, hit) && inRect(U.m, gClip);
    if (over && U.pressedNow) U.active = id;
    bool changed = false;
    if (U.active == id && U.down) {
        double t = clampd((U.m.x - tr.x) / tr.width, 0, 1);
        double nv = logS ? lo * std::exp(t * std::log(hi / lo)) : lo + t * (hi - lo);
        if (nv != v) { v = nv; changed = true; }
    }
    double t = logS ? std::log(clampd(v, lo, hi) / lo) / std::log(hi / lo) : (clampd(v, lo, hi) - lo) / (hi - lo);
    DrawRectangleRounded(tr, 1.0f, 4, Color{50, 60, 85, 255});
    DrawRectangleRounded(Rectangle{tr.x, tr.y, (float)(tr.width * t), tr.height}, 1.0f, 4, Color{80, 140, 235, 255});
    DrawCircleV(Vector2{(float)(tr.x + tr.width * t), tr.y + tr.height * 0.5f}, Sc(7.5f), (U.active == id || over) ? Color{255, 255, 255, 255} : Color{205, 220, 245, 255});
    return changed;
}
static bool uiCheck(Rectangle r, const char* label, bool& v) {
    int id = ++U.id;
    bool over = inRect(U.m, r) && inRect(U.m, gClip);
    if (over && U.pressedNow) U.pressed = id;
    bool click = over && U.released && U.pressed == id;
    if (click) v = !v;
    Rectangle bx = {r.x, r.y + Sc(2), Sc(17), Sc(17)};
    DrawRectangleRounded(bx, 0.25f, 4, over ? Color{58, 70, 98, 255} : Color{32, 40, 60, 255});
    DrawRectangleLinesEx(bx, 1.0f, Color{140, 165, 215, 160});
    if (v) DrawRectangleRounded(Rectangle{bx.x + 4, bx.y + 4, bx.width - 8, bx.height - 8}, 0.3f, 4, Color{90, 160, 255, 255});
    txt(label, r.x + Sc(25), r.y + Sc(2), Sc(14), Color{220, 228, 245, 255});
    return click;
}

// ---------------- acciones sobre cuerpos ----------------
static void setTilt(int i, double tiltDeg) {
    Body& b = A.S.b[i];
    V3 h{0, 0, 1}, r, v, c; double mu; Mat3 F;
    if (b.kind != K_STAR && A.S.relState(i, r, v, mu, c, F)) h = vnorm(cross(r, v));
    V3 e1 = cross(h, V3{0, 1, 0}); if (vlen(e1) < 1e-6) e1 = cross(h, V3{1, 0, 0});
    e1 = vnorm(e1); V3 e2 = cross(h, e1);
    double az = std::atan2(dot(b.spin, e2), dot(b.spin, e1)), t = tiltDeg * kDEG;
    b.spin = vnorm(h * std::cos(t) + (e1 * std::cos(az) + e2 * std::sin(az)) * std::sin(t));
}
static double currentTilt(int i) {
    Body& b = A.S.b[i];
    V3 h{0, 0, 1}, r, v, c; double mu; Mat3 F;
    if (b.kind != K_STAR && A.S.relState(i, r, v, mu, c, F)) h = vnorm(cross(r, v));
    return std::acos(clampd(dot(vnorm(b.spin), h), -1, 1)) / kDEG;
}
static void restoreBody(int i) {
    Body& b = A.S.b[i];
    b.gm = b.gm0; b.radiusKm = b.radius0; b.flat = b.flat0; b.spin = b.spin0; b.spinRate = b.spinRate0;
    A.S.recomputeMasses(); A.S.restoreOrbit(i); clearTrail(i); updatePeriods(); say("Valores originales restaurados");
}
static void deleteBody(int i) {
    if (i <= 0) { say("El Sol no se puede eliminar"); return; }
    removeBody(A.S, i); A.vo.selected = -1; if (A.focus == i) focusOn(0, true); updatePeriods(); say("Cuerpo eliminado");
}

// ---------------- panel de edicion ----------------
static void editPanel(Rectangle P) {
    Sim& S = A.S; int i = A.vo.selected;
    panelBox(P); gClip = P;
    BeginScissorMode((int)P.x, (int)P.y, (int)P.width, (int)P.height);
    float x = P.x + Sc(14), w = P.width - Sc(28), y = P.y + Sc(10) - A.scrollR, rowH = Sc(46);
    auto row = [&]() { Rectangle r = {x, y, w, Sc(40)}; y += rowH; return r; };
    if (i < 0 || i >= (int)S.b.size() || !S.b[i].alive) {
        txt("EDITAR CUERPO", x, y, Sc(17), Color{255, 255, 255, 255}); y += Sc(28);
        txt("Toca o haz clic en un planeta, luna,", x, y, Sc(14), Color{185, 198, 225, 255}); y += Sc(19);
        txt("cometa o asteroide para seleccionarlo", x, y, Sc(14), Color{185, 198, 225, 255}); y += Sc(19);
        txt("y editar masa, radio, órbita, rotación...", x, y, Sc(14), Color{185, 198, 225, 255}); y += Sc(34);
    } else {
        Body& b = S.b[i];
        const char* kind = b.kind == K_STAR ? "Estrella" : b.kind == K_PLANET ? (b.dwarf ? "Planeta enano" : "Planeta") : b.kind == K_MOON ? "Luna" : b.kind == K_COMET ? "Cometa" : "Asteroide / enano";
        txt(b.name.c_str(), x, y, Sc(19), Color{255, 255, 255, 255});
        txt(kind, x + w - txtW(kind, Sc(13)), y + Sc(5), Sc(13), Color{150, 190, 255, 255}); y += Sc(28);
        V3 r, v, cen; double mu = 0; Mat3 F; Elems el; bool hasOrb = S.relState(i, r, v, mu, cen, F);
        if (hasOrb) el = elementsFromState(mulMT(F, r), mulMT(F, v), mu);
        if (hasOrb) {
            double dist = vlen(r), spd = vlen(v) * AU_KM / 86400.0;
            const char* par = b.kind == K_MOON ? S.b[b.parent].name.c_str() : "Sol";
            txt(fmt("Distancia a %s: %s", par, dist < 0.02 ? fmt("%.0f km", dist * AU_KM) : fmt("%.4f UA", dist)), x, y, Sc(13), Color{200, 210, 230, 255}); y += Sc(17);
            txt(fmt("Velocidad %.2f km/s   Periodo %s", spd, el.bound ? (el.T > 730 ? fmt("%.2f años", el.T / 365.25) : fmt("%.2f días", el.T)) : "(no ligada)"), x, y, Sc(13), Color{200, 210, 230, 255}); y += Sc(17);
        }
        txt(fmt("Día: %.2f h   Inclinación axial: %.1f deg", 2 * kPI / std::fabs(b.spinRate + 1e-12) * 24.0, currentTilt(i)), x, y, Sc(13), Color{200, 210, 230, 255}); y += Sc(24);
        // masa
        if (!b.massless) {
            bool star = (b.kind == K_STAR); double unit = star ? b.gm0 : gmEarth(), val = b.gm / unit;
            double lo = star ? 0.05 : 1e-9, hi = star ? 30.0 : 1e5;
            const char* vt = star ? fmt("%.3f x Sol", val) : fmt("%.4g x Tierra", val);
            if (uiSlider(row(), "Masa", vt, val, lo, hi, true)) { b.gm = val * unit; S.recomputeMasses(); }
        }
        double rad = b.radiusKm;
        if (uiSlider(row(), "Radio", fmt("%.0f km", rad), rad, 1.0, 2.0e6, true)) b.radiusKm = rad;
        if (hasOrb && el.bound) {
            bool inKm = (b.kind == K_MOON); double a = el.a, e = el.e, inc = el.inc / kDEG, aShow = inKm ? a * AU_KM : a;
            double aLo = inKm ? std::max(S.b[b.parent].radiusKm * 1.3, 50.0) : 0.05, aHi = inKm ? 2.0e7 : 800.0;
            bool ch = uiSlider(row(), "Semieje mayor (órbita)", inKm ? fmt("%.0f km", aShow) : fmt("%.3f UA", aShow), aShow, aLo, aHi, true);
            double eSl = e; ch |= uiSlider(row(), "Excentricidad", fmt("%.3f", eSl), eSl, 0.0, 0.95, false);
            double iSl = inc; ch |= uiSlider(row(), "Inclinación orbital", fmt("%.2f deg", iSl), iSl, 0.0, 180.0, false);
            if (ch) { S.applyElements(i, inKm ? aShow / AU_KM : aShow, eSl, iSl * kDEG); clearTrail(i); updatePeriods(); }
        }
        // rotación
        bool locked = (b.kind == K_MOON && b.tidalLock);
        if (locked) { txt("Rotación sincrónica con su orbita", x, y, Sc(13), Color{185, 198, 225, 255}); y += Sc(22); }
        else {
            double perH = 2 * kPI / std::max(std::fabs(b.spinRate), 1e-9) * 24.0;
            if (uiSlider(row(), "Período de rotación", fmt("%.2f h", perH), perH, 0.1, 5000.0, true)) b.spinRate = (b.spinRate < 0 ? -1.0 : 1.0) * 2 * kPI / (perH / 24.0);
            bool retro = b.spinRate < 0; Rectangle cr = {x, y - Sc(2), w, Sc(22)};
            if (uiCheck(cr, "Rotación retrograda", retro)) b.spinRate = (retro ? -1.0 : 1.0) * std::fabs(b.spinRate);
            y += Sc(28);
            double tilt = currentTilt(i);
            if (uiSlider(row(), "Inclinación del eje", fmt("%.1f deg", tilt), tilt, 0.0, 180.0, false)) setTilt(i, tilt);
        }
        if (b.kind == K_MOON && uiCheck(Rectangle{x, y - Sc(2), w, Sc(22)}, "Rotación sincrónica", b.tidalLock)) {}
        if (b.kind == K_MOON) y += Sc(28);
        // botones
        float bw = (w - Sc(8)) / 2, bh = Sc(32);
        if (uiButton(Rectangle{x, y, bw, bh}, "Circularizar órbita", false, hasOrb && b.kind != K_STAR)) { S.circularize(i); clearTrail(i); updatePeriods(); say("Órbita circularizada"); }
        if (uiButton(Rectangle{x + bw + Sc(8), y, bw, bh}, "Restaurar", false)) restoreBody(i);
        y += bh + Sc(8);
        if (uiButton(Rectangle{x, y, bw, bh}, "Cambiar aspecto", false)) {
            int& ls = A.lookSel[i]; ls = (ls + 1) % (int)presets().size(); Look old = b.look; b.look = presets()[ls].look;
            if (old.ringOut > 0 && b.look.ringOut == 0) {}
            say(fmt("Aspecto: %s", presets()[ls].name));
        }
        if (uiButton(Rectangle{x + bw + Sc(8), y, bw, bh}, "Enfocar camara", false)) focusOn(i, true);
        y += bh + Sc(8);
        if (uiButton(Rectangle{x, y, w, bh}, "Eliminar cuerpo", false, b.kind != K_STAR)) deleteBody(i);
        y += bh + Sc(14);
    }
    // anadir
    txt("AÑADIR", x, y, Sc(15), Color{255, 255, 255, 255}); y += Sc(24);
    float bw3 = (w - Sc(12)) / 3, bh = Sc(32);
    if (uiButton(Rectangle{x, y, bw3, bh}, "+ Planeta")) {
        int nb = addUserPlanet(S, A.rng.range(1.3, 2.6), (int)(A.rng.next() % presets().size()), A.rng); ensureTrails(); updatePeriods();
        A.vo.selected = nb; focusOn(nb, true); say("Planeta creado: usa los controles para editarlo");
    }
    int pb = (i > 0 && i < (int)S.b.size()) ? (S.b[i].kind == K_MOON ? S.b[i].parent : (S.b[i].kind == K_PLANET ? i : -1)) : -1;
    if (uiButton(Rectangle{x + bw3 + Sc(6), y, bw3, bh}, "+ Luna", false, pb > 0)) {
        int nb = addUserMoon(S, pb, A.rng); ensureTrails(); updatePeriods(); if (nb > 0) { A.vo.selected = nb; focusOn(nb, true); say("Luna creada"); }
    }
    if (uiButton(Rectangle{x + 2 * (bw3 + Sc(6)), y, bw3, bh}, "+ Cometa")) {
        int nb = addUserComet(S, A.rng); ensureTrails(); updatePeriods(); A.vo.selected = nb; focusOn(nb, true); say("Cometa creado");
    }
    y += bh + Sc(10);
    EndScissorMode(); gClip = Rectangle{0, 0, 100000, 100000};
    float content = y + A.scrollR - P.y; float maxScroll = std::max(0.0f, content - P.height + Sc(10));
    if (inRect(U.m, P)) A.scrollR = (float)clampd(A.scrollR - GetMouseWheelMove() * Sc(40), 0, maxScroll);
    A.scrollR = std::min(A.scrollR, maxScroll);
}

// ---------------- lista de cuerpos ----------------
static void listPanel(Rectangle P) {
    Sim& S = A.S; panelBox(P); gClip = P;
    BeginScissorMode((int)P.x, (int)P.y, (int)P.width, (int)P.height);
    float x = P.x + Sc(12), y = P.y + Sc(8) - A.scrollL, rh = Sc(23);
    txt("CUERPOS", x, y, Sc(15), Color{255, 255, 255, 255}); y += Sc(24);
    int sel = A.vo.selected, selPar = (sel > 0 && S.b[sel].kind == K_MOON) ? S.b[sel].parent : -1;
    for (size_t i = 0; i < S.b.size(); i++) {
        const Body& b = S.b[i];
        if (!b.alive) continue;
        bool moon = (b.kind == K_MOON);
        if (moon && !((int)b.parent == sel || (int)b.parent == selPar || b.parent == A.focus)) continue;
        Rectangle rr = {P.x + Sc(6), y, P.width - Sc(12), rh};
        bool over = inRect(U.m, rr) && inRect(U.m, P);
        if ((int)i == sel) DrawRectangleRounded(rr, 0.3f, 4, Color{60, 112, 200, 150});
        else if (over) DrawRectangleRounded(rr, 0.3f, 4, Color{60, 70, 100, 150});
        if (over && U.pressedNow) {
            double now = GetTime();
            if (A.lastClickBody == (int)i && now - A.lastClickT < 0.4) focusOn((int)i, true);
            A.vo.selected = (int)i; A.lastClickT = now; A.lastClickBody = (int)i;
        }
        float ind = moon ? Sc(18) : 0;
        DrawCircleV(Vector2{x + ind + Sc(5), y + rh * 0.5f}, Sc(4.5f), b.col);
        txt(b.name.c_str(), x + ind + Sc(16), y + Sc(3), Sc(14), moon ? Color{190, 200, 220, 255} : Color{235, 240, 250, 255});
        y += rh;
    }
    y += Sc(10);
    txt("Opciones de imagen", x, y, Sc(13), Color{150, 190, 255, 255}); y += Sc(22);
    double ex = A.vo.exposure, am = A.vo.ambient * 100.0;
    if (uiSlider(Rectangle{x, y, P.width - Sc(24), Sc(40)}, "Exposición", fmt("%.2f", ex), ex, 0.3, 3.0, true)) A.vo.exposure = (float)ex;
    y += Sc(46);
    if (uiSlider(Rectangle{x, y, P.width - Sc(24), Sc(40)}, "Luz ambiente", fmt("%.1f", am), am, 0.0, 12.0, false)) A.vo.ambient = (float)(am / 100.0);
    y += Sc(50);
    EndScissorMode(); gClip = Rectangle{0, 0, 100000, 100000};
    float content = y + A.scrollL - P.y, maxScroll = std::max(0.0f, content - P.height);
    if (inRect(U.m, P)) A.scrollL = (float)clampd(A.scrollL - GetMouseWheelMove() * Sc(40), 0, maxScroll);
    A.scrollL = std::min(A.scrollL, maxScroll);
}

// ---------------- barra de herramientas, HUD, etiquetas, ayuda ----------------
struct Layout { Rectangle top, left, right, bar; };
static Layout layoutUI(int W, int H) {
    float s = U.s, barW = std::min((float)W - 20 * s, 940 * s), barH = 96 * s, topH = 112 * s, leftW = 236 * s, rightW = 334 * s;
    Layout L;
    L.top = {10 * s, 10 * s, 430 * s, topH};
    L.bar = {((float)W - barW) * 0.5f, (float)H - barH - 10 * s, barW, barH};
    L.left = {10 * s, topH + 20 * s, leftW, std::max(120.0f, (float)H - topH - barH - 50 * s)};
    L.right = {(float)W - rightW - 10 * s, 10 * s, rightW, std::max(160.0f, (float)H - barH - 30 * s)};
    return L;
}
static void toolbar(Rectangle R) {
    panelBox(R);
    float pad = Sc(9), h = Sc(34), gap = Sc(6), x = R.x + pad, y = R.y + pad;
    auto btn = [&](const char* label, float w, bool active = false) { Rectangle r = {x, y, w, h}; x += w + gap; return uiButton(r, label, active); };
    if (btn("Ahora", Sc(66))) { rebuild(nowJD()); say("Fecha y hora actuales"); }
    if (btn("-1 año", Sc(66))) { rebuild(A.jdStart + A.S.t - 365.25); say("Un año atrás (se reinicia el modelo)"); }
    if (btn("+1 año", Sc(66))) { rebuild(A.jdStart + A.S.t + 365.25); say("Un año adelante (se reinicia el modelo)"); }
    x += gap * 2;
    if (btn("<<", Sc(44))) A.speedIdx = std::max(0, A.speedIdx - 1);
    if (btn(A.playing ? "Pausa" : "Reanudar", Sc(88), !A.playing)) A.playing = !A.playing;
    if (btn(">>", Sc(44))) A.speedIdx = std::min(kNumSpeeds - 1, A.speedIdx + 1);
    if (btn(A.dir > 0 ? "Adelante" : "Atrás", Sc(80), A.dir < 0)) { A.dir = -A.dir; clearTrails(); }
    txt(fmt("%s%s", A.dir < 0 ? "-" : "", fmtSpeed(kSpeeds[A.speedIdx]).c_str()), x + gap, y + Sc(8), Sc(15), Color{255, 230, 150, 255});
    x = R.x + pad; y += h + Sc(8);
    ViewOpts& v = A.vo;
    if (btn("Órbitas", Sc(74), v.orbits)) v.orbits = !v.orbits;
    if (btn("Estelas", Sc(68), v.trails)) v.trails = !v.trails;
    if (btn("Etiquetas", Sc(80), v.labels)) v.labels = !v.labels;
    if (btn("Cuadrícula", Sc(86), v.grid)) v.grid = !v.grid;
    if (btn("Ejes", Sc(54), v.axes)) v.axes = !v.axes;
    if (btn("Asteroides", Sc(88), v.swarm)) v.swarm = !v.swarm;
    if (btn("Atmósfera", Sc(84), v.atmos)) v.atmos = !v.atmos;
    if (btn("Órb. lunas", Sc(88), v.moonOrbits)) v.moonOrbits = !v.moonOrbits;
    if (btn(A.panMode ? "Mover" : "Orbitar", Sc(70), A.panMode)) A.panMode = !A.panMode;
    if (btn("Ayuda", Sc(58), A.showHelp)) A.showHelp = !A.showHelp;
}
static void hudPanel(Rectangle R) {
    panelBox(R);
    float x = R.x + Sc(14), y = R.y + Sc(9);
    double jd = A.jdStart + A.S.t; int Y, M, D; double hr; dateFromJd(jd, Y, M, D, hr);
    int hh = (int)hr, mm = (int)((hr - hh) * 60.0), ss = (int)(((hr - hh) * 60.0 - mm) * 60.0);
    txt(fmt("%02d %s %d   %02d:%02d:%02d UTC", D, kMeses[(M - 1 + 12) % 12], Y, hh, mm, ss), x, y, Sc(21), Color{255, 255, 255, 255}); y += Sc(27);
    txt(fmt("Fecha juliana %.3f     %.0f FPS", jd, A.fps), x, y, Sc(13), Color{160, 175, 205, 255}); y += Sc(19);
    std::string sp = (A.playing ? "" : "PAUSA  ") + std::string(A.dir < 0 ? "-" : "") + fmtSpeed(kSpeeds[A.speedIdx]);
    if (A.playing && A.limited) sp += fmt("   (limitada por CPU: %.3g d/s)", A.achieved);
    txt(sp.c_str(), x, y, Sc(14), A.limited ? Color{255, 170, 110, 255} : Color{255, 230, 150, 255}); y += Sc(19);
    const Body& fb = A.S.b[A.focus];
    double dist = std::exp(A.logDist), kmpx = dist * AU_KM * 2.0 * A.fc.tanHalf / std::max(1, A.fc.H);
    txt(fmt("Foco: %s   Distancia %.4g UA   1 px = %.4g km", fb.name.c_str(), dist, kmpx), x, y, Sc(13), Color{200, 215, 240, 255}); y += Sc(18);
    if (A.degraded) txt("Integración adaptativa: precisión reducida a esta velocidad", x, y, Sc(12), Color{255, 170, 110, 255});
}
static void drawOverlay(const Layout& L) {
    Sim& S = A.S; FrameCtx& f = A.fc;
    A.scr.assign(S.b.size(), Vector2{-1e9f, -1e9f});
    for (size_t i = 0; i < S.b.size(); i++) {
        const Body& b = S.b[i];
        if (!b.alive) continue;
        Vector2 p; double z;
        if (!projectRel(f, relMm(f, b.pos), p, z)) continue;
        A.scr[i] = p;
        double pxR = (i < A.pxr.size()) ? A.pxr[i] : 0;
        bool sel = ((int)i == A.vo.selected), foc = ((int)i == A.focus);
        if (p.x < -50 || p.y < -50 || p.x > f.W + 50 || p.y > f.H + 50) continue;
        float rad = (float)std::max(pxR, 0.0);
        if (rad < Sc(3.2f)) {
            Color c = b.col; c.a = (b.kind == K_MOON ? 150 : 215);
            DrawCircleV(p, Sc(b.kind == K_STAR ? 3.4f : 2.6f), c);
            if (b.kind != K_MOON && b.kind != K_STAR) DrawCircleLines((int)p.x, (int)p.y, Sc(5.2f), Color{b.col.r, b.col.g, b.col.b, 90});
        }
        if (sel) {
            float r = std::max(rad + Sc(8), Sc(14));
            DrawCircleLines((int)p.x, (int)p.y, r, Color{255, 255, 255, 230});
            float q = r * 0.55f;
            DrawLineEx(Vector2{p.x - r - q, p.y}, Vector2{p.x - r, p.y}, 1.5f, WHITE); DrawLineEx(Vector2{p.x + r, p.y}, Vector2{p.x + r + q, p.y}, 1.5f, WHITE);
            DrawLineEx(Vector2{p.x, p.y - r - q}, Vector2{p.x, p.y - r}, 1.5f, WHITE); DrawLineEx(Vector2{p.x, p.y + r}, Vector2{p.x, p.y + r + q}, 1.5f, WHITE);
        }
        bool show = A.vo.labels || sel || foc;
        if (show && b.kind == K_MOON && !sel && !foc) {
            double pr = (b.parent < (int)A.pxr.size()) ? A.pxr[b.parent] : 0;
            Vector2 pp = A.scr[b.parent];
            double sepPx = std::hypot(p.x - pp.x, p.y - pp.y);
            show = (pr > 7 && sepPx > Sc(22)) || (pr <= 7 && false);
        }
        if (show) {
            float fs = Sc(b.kind == K_MOON ? 13.0f : 16.0f);
            Color c = sel ? Color{255, 255, 255, 255} : (b.kind == K_MOON ? Color{190, 205, 235, 210} : Color{235, 242, 255, 235});
            txt(b.name.c_str(), p.x + std::max(rad, Sc(3.5f)) + Sc(7), p.y - fs * 0.6f, fs, c);
        }
    }
    if (A.vo.grid) {   // rotulos de la cuadricula
        static const double R[] = {0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200, 500};
        V3 sp = S.b[0].pos; double dS = vlen(sp - f.cam) * f.mm + 1e-9;
        for (double r : R) {
            double px = (r * f.mm) / dS * f.focalPx;
            if (px < 40 || px > 30000) continue;
            for (int k = 0; k < 24; k++) {
                double a = 2 * kPI * (k + 0.5) / 24;
                Vector2 p; double z;
                if (!projectRel(f, relMm(f, sp + V3{std::cos(a) * r, std::sin(a) * r, 0}), p, z)) continue;
                if (p.x > Sc(260) && p.x < f.W - Sc(350) && p.y > Sc(130) && p.y < f.H - Sc(120)) { txt(fmt("%g UA", r), p.x + 4, p.y - 8, Sc(12), Color{140, 170, 230, 150}); break; }
            }
        }
    }
}
static void helpOverlay(int W, int H) {
    float w = std::min((float)W - Sc(40), Sc(760)), h = Sc(470);
    Rectangle R = {((float)W - w) * 0.5f, ((float)H - h) * 0.5f - Sc(20), w, h};
    DrawRectangleRounded(R, 0.03f, 8, Color{6, 10, 20, 240}); DrawRectangleLinesEx(R, 1.0f, Color{120, 150, 220, 130});
    float x = R.x + Sc(22), y = R.y + Sc(16), lh = Sc(19);
    auto head = [&](const char* s) { y += Sc(5); txt(s, x, y, Sc(15), Color{120, 180, 255, 255}); y += Sc(22); };
    auto line = [&](const char* s) { txt(s, x + Sc(8), y, Sc(14), Color{225, 232, 248, 255}); y += lh; };
    txt("SIMULADOR DEL SISTEMA SOLAR", x, y, Sc(22), WHITE); y += Sc(30);
    txt("N cuerpos con lunas, asteroides y cometas · C++ / Raylib · física en UA y días", x, y, Sc(13), Color{160, 175, 205, 255}); y += Sc(24);
    head("CÁMARA");
    line("Arrastrar (botón izq. / un dedo): orbitar      Rueda / pellizcar: acercar y alejar");
    line("Botón der. o Mayús+arrastrar: desplazar      Doble clic en un cuerpo: enfocarlo");
    line("Flechas: orbitar      + / - o Re Pág / Av Pág: zoom      Inicio: vista general      0-9: Sol, planetas, Luna");
    head("TIEMPO");
    line("Espacio: pausa      , y . : más lento / más rápido      R: invertir el tiempo      N: ahora (fecha real)");
    head("VISTA");
    line("O órbitas   T estelas   L etiquetas   G cuadrícula   E ejes y ecuador   A asteroides   V atmósferas   M órbitas de lunas");
    line("Tab lista de cuerpos   U ocultar interfaz   K densidad de asteroides   F11 pantalla completa   F12 captura   F9 / F10 tamaño de la interfaz");
    head("EDICIÓN");
    line("Haz clic en un cuerpo y usa el panel derecho: masa, radio, órbita (a, e, i), rotación, inclinación del eje y aspecto.");
    line("+ Planeta / + Luna / + Cometa los crea al instante; el N-cuerpos recalcula las perturbaciones.");
    line("C circularizar órbita      F enfocar      Supr eliminar      H o F1 muestra / oculta esta ayuda");
    txt("Pulsa H para cerrar", R.x + R.width - txtW("Pulsa H para cerrar", Sc(13)) - Sc(18), R.y + R.height - Sc(26), Sc(13), Color{150, 170, 210, 255});
}

// ---------------- entrada ----------------
static int pickBody(Vector2 m) {
    int best = -1; double bs = 1e30;
    for (size_t i = 0; i < A.S.b.size() && i < A.scr.size(); i++) {
        if (!A.S.b[i].alive || A.scr[i].x < -1e8f) continue;
        double d = std::hypot(m.x - A.scr[i].x, m.y - A.scr[i].y), rad = std::max(i < A.pxr.size() ? A.pxr[i] : 0.0, (double)Sc(13));
        if (d <= rad && d / rad < bs) { bs = d / rad; best = (int)i; }
    }
    return best;
}
static void resetView() {
    A.yaw = 0.7; A.pitch = 0.55; A.fromPos = A.fpos; A.fromLog = A.logDist; A.focus = 0; A.pan = V3{}; A.toLog = std::log(7.0); A.trans = true; A.tt = 0;
}
static void handleKeys(float dt) {
    Sim& S = A.S; ViewOpts& v = A.vo;
    if (IsKeyPressed(KEY_SPACE)) A.playing = !A.playing;
    if (IsKeyPressed(KEY_PERIOD)) A.speedIdx = std::min(kNumSpeeds - 1, A.speedIdx + 1);
    if (IsKeyPressed(KEY_COMMA)) A.speedIdx = std::max(0, A.speedIdx - 1);
    if (IsKeyPressed(KEY_R)) { A.dir = -A.dir; clearTrails(); }
    if (IsKeyPressed(KEY_N)) { rebuild(nowJD()); say("Fecha y hora actuales"); }
    if (IsKeyPressed(KEY_K)) { A.swarmLevel = (A.swarmLevel + 1) % 5; applySwarmLevel(); rebuild(A.jdStart + A.S.t); say(fmt("Densidad de asteroides: nivel %d de 4 (se reinicia el modelo)", A.swarmLevel)); }
    if (IsKeyPressed(KEY_O)) v.orbits = !v.orbits;
    if (IsKeyPressed(KEY_T)) v.trails = !v.trails;
    if (IsKeyPressed(KEY_L)) v.labels = !v.labels;
    if (IsKeyPressed(KEY_G)) v.grid = !v.grid;
    if (IsKeyPressed(KEY_E)) v.axes = !v.axes;
    if (IsKeyPressed(KEY_A)) v.swarm = !v.swarm;
    if (IsKeyPressed(KEY_V)) v.atmos = !v.atmos;
    if (IsKeyPressed(KEY_M)) v.moonOrbits = !v.moonOrbits;
    if (IsKeyPressed(KEY_U)) A.showUI = !A.showUI;
    if (IsKeyPressed(KEY_TAB)) A.showList = !A.showList;
    if (IsKeyPressed(KEY_H) || IsKeyPressed(KEY_F1)) A.showHelp = !A.showHelp;
    if (IsKeyPressed(KEY_HOME)) resetView();
    if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
    if (IsKeyPressed(KEY_F12)) { TakeScreenshot(fmt("sistema_solar_%d.png", (int)GetTime())); say("Captura guardada en la carpeta del programa"); }
    if (IsKeyPressed(KEY_F9)) A.uiUser = std::max(0.7f, A.uiUser - 0.1f);
    if (IsKeyPressed(KEY_F10)) A.uiUser = std::min(2.2f, A.uiUser + 0.1f);
    if (IsKeyPressed(KEY_F) && v.selected >= 0) focusOn(v.selected, true);
    if (IsKeyPressed(KEY_C) && v.selected > 0) { S.circularize(v.selected); clearTrail(v.selected); updatePeriods(); say("Órbita circularizada"); }
    if (IsKeyPressed(KEY_DELETE) && v.selected > 0) deleteBody(v.selected);
    static const char* digit[10] = {"sol", "mercurio", "venus", "tierra", "marte", "jupiter", "saturno", "urano", "neptuno", "luna"};
    static const int dk[10] = {KEY_ZERO, KEY_ONE, KEY_TWO, KEY_THREE, KEY_FOUR, KEY_FIVE, KEY_SIX, KEY_SEVEN, KEY_EIGHT, KEY_NINE};
    for (int k = 0; k < 10; k++) if (IsKeyPressed(dk[k])) { int i = findKey(digit[k]); if (i >= 0) { v.selected = i; focusOn(i, true); } }
    float rot = 1.5f * dt;
    if (IsKeyDown(KEY_LEFT)) A.yaw += rot;
    if (IsKeyDown(KEY_RIGHT)) A.yaw -= rot;
    if (IsKeyDown(KEY_UP)) A.pitch += rot;
    if (IsKeyDown(KEY_DOWN)) A.pitch -= rot;
    if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD) || IsKeyDown(KEY_PAGE_UP)) { A.logDist -= 1.4 * dt; A.trans = false; }
    if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT) || IsKeyDown(KEY_PAGE_DOWN)) { A.logDist += 1.4 * dt; A.trans = false; }
}
static void cameraInput(bool overUI) {
    Vector2 md = GetMouseDelta();
    static float pinchPrev = 0; static Vector2 midPrev{0, 0};
    int tc = GetTouchPointCount();
    double perPx = std::exp(A.logDist) * 2.0 * A.fc.tanHalf / std::max(1, A.fc.H);
    if (tc >= 2) {
        Vector2 a = GetTouchPosition(0), b = GetTouchPosition(1);
        float d = std::hypot(b.x - a.x, b.y - a.y); Vector2 mid = {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
        if (pinchPrev > 1 && d > 1) { A.logDist -= std::log(d / pinchPrev); A.trans = false; A.pan += (A.fc.right * (-(mid.x - midPrev.x)) + A.fc.up * (mid.y - midPrev.y)) * perPx; }
        pinchPrev = d; midPrev = mid; A.dragScene = false;
        return;
    }
    pinchPrev = 0;
    if (A.dragScene) {
        bool pan = IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) || IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT) || A.panMode;
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            if (pan) A.pan += (A.fc.right * (-md.x) + A.fc.up * md.y) * perPx;
            else { A.yaw -= md.x * 0.0052; A.pitch += md.y * 0.0052; }
        }
    }
    if (!overUI) {
        float w = GetMouseWheelMove();
        if (w != 0) { A.logDist -= w * 0.12; A.trans = false; }
    }
}

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 900, "Simulador del Sistema Solar - N cuerpos (C++ / Raylib)");
    SetWindowMinSize(900, 560);
    {
        int mw = GetMonitorWidth(0), mh = GetMonitorHeight(0);
        if (mw > 0 && mh > 0) { int w = std::min(1600, mw - 60), h = std::min(900, mh - 110); SetWindowSize(w, h); SetWindowPosition(std::max(0, (mw - w) / 2), std::max(24, (mh - h) / 2 - 16)); }
    }
    SetTargetFPS(120);
    gfxInit(A.g);
    applySwarmLevel();
    rebuild(nowJD());
    A.fpos = A.S.b[0].pos; A.logDist = std::log(7.0); A.showHelp = true;
    if (!A.g.status.empty()) say(A.g.status);
    int lastDir = 1; double perT = 0;
    while (!WindowShouldClose()) {
        float dt = std::min(GetFrameTime(), 0.1f);
        int W = GetScreenWidth(), H = GetScreenHeight();
        U.s = (float)clampd(H / 900.0, 0.8, 2.2) * A.uiUser;
        U.id = 0; U.m = GetMousePosition();
        U.pressedNow = IsMouseButtonPressed(MOUSE_BUTTON_LEFT); U.down = IsMouseButtonDown(MOUSE_BUTTON_LEFT); U.released = IsMouseButtonReleased(MOUSE_BUTTON_LEFT);
        if (U.pressedNow) U.pressed = -1;
        A.fps = A.fps * 0.9f + GetFPS() * 0.1f;
        Layout L = layoutUI(W, H);
        bool panelOver = A.showUI && (inRect(U.m, L.top) || inRect(U.m, L.bar) || inRect(U.m, L.right) || (A.showList && inRect(U.m, L.left)));
        if (A.showHelp && U.pressedNow && !panelOver) A.showHelp = false;
        bool overUI = panelOver || U.active >= 0 || (A.showHelp && A.showUI);
        ensureTrails();
        // ---- entrada
        handleKeys(dt);
        if (U.pressedNow && !overUI) { A.dragScene = true; A.pressPos = U.m; }
        cameraInput(overUI);
        if (U.released) {
            if (A.dragScene && std::hypot(U.m.x - A.pressPos.x, U.m.y - A.pressPos.y) < 6.0f && GetTouchPointCount() < 2) {
                int hit = pickBody(U.m); double now = GetTime();
                if (hit >= 0) { if (hit == A.lastClickBody && now - A.lastClickT < 0.4) focusOn(hit, true); A.vo.selected = hit; A.lastClickT = now; A.lastClickBody = hit; }
                else A.vo.selected = -1;
            }
            A.dragScene = false;
        }
        // ---- fisica
        if (A.dir != lastDir) { clearTrails(); lastDir = A.dir; }
        double req = A.playing ? A.dir * kSpeeds[A.speedIdx] / 86400.0 * dt : 0.0, done = 0; bool trunc = false;
        auto t0 = std::chrono::steady_clock::now();
        if (req != 0.0) A.S.advance(req, 0.011, done, trunc);
        double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (Body& b : A.S.b) {
            if (!b.alive) continue;
            double dphi = b.spinRate * done;
            b.spinPhase = std::fmod(b.spinPhase + dphi, 2 * kPI);
            if (std::fabs(dphi) <= 0.35) b.dispPhase = b.spinPhase; else b.dispPhase += (dphi > 0 ? 0.35 : -0.35);
        }
        A.limited = trunc; A.achieved = (dt > 0 && req != 0.0) ? std::fabs(done) / dt : 0.0;
        if (req != 0.0) {   // regulador de rendimiento: sacrifica precision de las lunas antes que fluidez
            double md = A.S.moonDiv, hm = A.S.Hmax;
            if (trunc) { md = std::max(8.0, md * 0.9); hm = std::min(2.0, hm * 1.12); }
            else if (tp < 0.0045) { md = std::min(60.0, md * 1.04); hm = std::max(0.5, hm * 0.96); }
            if (std::fabs(md - A.S.moonDiv) > 0.8 || std::fabs(hm - A.S.Hmax) > 0.03) { A.S.moonDiv = md; A.S.Hmax = hm; A.S.recomputeMasses(); }
            A.degraded = (A.S.moonDiv < 52 || A.S.Hmax > 0.6);
        }
        perT += dt; if (perT > 1.5) { perT = 0; updatePeriods(); }
        // ---- camara
        if (!A.S.b[A.focus].alive) focusOn(0, true);
        FrameCtx& f = A.fc;
        f.W = W; f.H = H; f.tanHalf = std::tan(25.0 * kDEG); f.focalPx = (H * 0.5) / f.tanHalf;
        V3 target = A.S.b[A.focus].pos + A.pan;
        if (A.trans) {
            A.tt += dt / 1.5; double k = std::min(A.tt, 1.0); k = k * k * (3 - 2 * k);
            A.fpos = A.fromPos + (target - A.fromPos) * k; A.logDist = A.fromLog + (A.toLog - A.fromLog) * k;
            if (A.tt >= 1.0) A.trans = false;
        } else A.fpos = target;
        double minLog = std::log(std::max(A.S.b[A.focus].radiusKm / AU_KM * 1.12, 1e-10));
        A.logDist = clampd(A.logDist, minLog, std::log(3000.0)); A.pitch = clampd(A.pitch, -1.553, 1.553);
        double dist = std::exp(A.logDist);
        V3 dv = {std::cos(A.pitch) * std::cos(A.yaw), std::cos(A.pitch) * std::sin(A.yaw), std::sin(A.pitch)};
        f.cam = A.fpos + dv * dist; f.fwd = dv * -1.0; f.right = vnorm(cross(f.fwd, V3{0, 0, 1})); f.up = cross(f.right, f.fwd);
        A.vo.simDays = A.S.t;
        // ---- dibujo
        BeginDrawing();
        ClearBackground(BLACK);
        drawSky(A.g, f);
        render3D(A.g, f, A.S, A.trails, A.vo, A.pxr);
        drawOverlay(L);
        if (A.showUI) {
            hudPanel(L.top); toolbar(L.bar); editPanel(L.right);
            if (A.showList) listPanel(L.left);
            if (A.showHelp) helpOverlay(W, H);
        }
        if (GetTime() - A.toastT < 3.5 && !A.toast.empty()) {
            float al = (float)clampd((3.5 - (GetTime() - A.toastT)) * 2.0, 0, 1), tw = txtW(A.toast.c_str(), Sc(16));
            Rectangle tr = {((float)W - tw) * 0.5f - Sc(14), (float)H - Sc(150), tw + Sc(28), Sc(34)};
            DrawRectangleRounded(tr, 0.4f, 6, Color{20, 30, 55, (unsigned char)(230 * al)});
            txt(A.toast.c_str(), tr.x + Sc(14), tr.y + Sc(8), Sc(16), Color{255, 255, 255, (unsigned char)(255 * al)});
        }
        EndDrawing();
        if (U.released) U.pressed = -1;
        if (!U.down) U.active = -1;
    }
    CloseWindow();
    return 0;
}
#endif
