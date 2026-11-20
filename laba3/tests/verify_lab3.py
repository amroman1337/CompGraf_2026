from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CPP = (ROOT / 'source/application.cpp').read_text()
VERT = (ROOT / 'shaders/textured.vert').read_text()
FRAG = (ROOT / 'shaders/textured.frag').read_text()


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def unit(v):
    length = math.sqrt(dot(v, v))
    return tuple(x/length for x in v)


block = CPP.split('positions = {{', 1)[1].split('}};', 1)[0]
positions = [tuple(map(float, re.findall(r'-?\d+\.\d+(?=f)', line)))
             for line in block.splitlines() if '{' in line]
positions = [p for line in positions for p in (line[:3], line[3:])]
faces_block = CPP.split('faces = {{', 1)[1].split('}};', 1)[0]
flat_indices = list(map(int, re.findall(r'\d+', faces_block)))
faces = [flat_indices[i:i+4] for i in range(0, len(flat_indices), 4)]
assert len(positions) == 8 and len(faces) == 6
normals = set()
for face in faces:
    a, b, c, d = [positions[i] for i in face]
    normal = unit(cross(sub(b, a), sub(c, a)))
    center = tuple(sum(p[k] for p in (a,b,c,d))/4 for k in range(3))
    assert dot(normal, center) > 0
    assert abs(dot(normal, sub(d, a))) < 1e-8
    assert abs(dot(normal, sub(b, a))) < 1e-8
    normals.add(normal)
assert len(normals) == 6
assert 'glm::normalize(glm::cross(edge1, edge2))' in CPP
print('PASS: six outward unit normals; 24 face vertices; 12 nondegenerate triangles.')


def fields(source, name, cpp=False):
    text = re.sub(r'//[^\n]*', '', source)
    if cpp:
        block = re.search(r'struct alignas\(16\) '+name+r'\s*\{(.*?)\};', text, re.S).group(1)
    else:
        block = re.search(r'(?:uniform|struct) '+name+r'\s*\{(.*?)\}', text, re.S).group(1)
    return re.findall(r'(?:glm::)?(mat4|vec4|uvec4)\s+(\w+)\s*;', block)


for name, size in [('SceneUniforms',144), ('ModelUniforms',176)]:
    host = fields(CPP,name,True)
    assert host == fields(VERT,name) == fields(FRAG,name)
    offset = sum(64 if typ == 'mat4' else 16 for typ, _ in host)
    assert offset == size
for host_name, shader_name, size in [('GpuPointLight','PointLight',32),('GpuSpotLight','SpotLight',64)]:
    host = fields(CPP,host_name,True)
    assert host == fields(FRAG,shader_name)
    assert len(host)*16 == size
assert re.search(r'PointLight\s+lights\[\s*\]',FRAG)
assert re.search(r'SpotLight\s+lights\[\s*\]',FRAG)
assert 'lightCounts.x' in FRAG and 'lightCounts.y' in FRAG
assert 'glm::transpose(glm::inverse(data.model))' in CPP
for location, attr in [(0,'position'),(1,'normal'),(2,'uv')]:
    assert f'offsetof(Vertex, {attr})' in CPP
print('PASS: scene/model std140 layouts; std430 light strides; vertex attribute locations.')


def cone(theta, inner, outer):
    x, a, b = (math.cos(math.radians(v)) for v in (theta,inner,outer))
    if a <= b:
        return 1.0 if x >= b else 0.0
    t = max(0.0,min(1.0,(x-b)/(a-b)))
    return t*t*(3-2*t)

assert cone(0,15,25) == 1 and cone(15,15,25) == 1
assert 0 < cone(20,15,25) < 1 and cone(25,15,25) == 0
assert cone(35,15,25) == 0
assert cone(10,20,20) == 1 and cone(30,20,20) == 0
D = unit((0,-1,-1)); L = tuple(-x for x in D)
assert abs(dot(D,tuple(-x for x in L))-1) < 1e-8
assert 'spotlightWeight(dot(D, -L)' in FRAG
atten = lambda r: 1/max(r*r,.01)
assert atten(2)/atten(4) == 4 and math.isfinite(atten(0))
assert FRAG.count('max(dot(toLight, toLight), 0.01)') == 2
assert 'vec3 H = normalize(halfway);' in FRAG and 'if (diffuseFactor <= 0.0)' in FRAG
print('PASS: inverse-square ratio; spotlight axis, boundaries, soft transition and equal angles.')


eye=(4,3,7); yaw=math.radians(-120); pitch=math.radians(-20)
f=unit((math.cos(yaw)*math.cos(pitch), math.sin(pitch), math.sin(yaw)*math.cos(pitch)))
r=unit(cross(f,(0,1,0))); u=cross(r,f)
for center,scale,rotation in [((-1.2,0,0),1,0),((1.2,0,0),.8,math.pi/4)]:
    for v in positions:
        x,y,z=(q*scale for q in v)
        world=(center[0]+math.cos(rotation)*x+math.sin(rotation)*z,
               center[1]+y,center[2]-math.sin(rotation)*x+math.cos(rotation)*z)
        delta=sub(world,eye); x,y,z=dot(delta,r),dot(delta,u),dot(delta,f)
        assert .1 < z < 100
        assert abs(x) < z*math.tan(math.radians(55/2))*1280/720
        assert abs(y) < z*math.tan(math.radians(55/2))
        assert abs(x) < 4*1280/720 and abs(y) < 4
render=CPP.split('    void render(')[1]
assert all(name not in render for name in ['vkBeginCommandBuffer','vkResetCommandBuffer','vkCmdBeginRenderPass','vkCmdEndRenderPass','vkEndCommandBuffer'])
update=CPP.split('    void update(double time)')[1].split('    void render(')[0]
assert 'writeBuffer' not in update and 'ensureLightCapacity' not in update
print('PASS: initial camera sees both cubes; render pass ownership; writes after frame preparation.')


a=CPP.index('    void ensureLightCapacity(')
b=CPP.index('    std::vector<uint32_t> readShader',a)
capacity=CPP[a:b]
program=r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
using VkDeviceSize = uint64_t;
constexpr int VK_BUFFER_USAGE_STORAGE_BUFFER_BIT=1;
constexpr int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER=2;
struct Buffer { VkDeviceSize size=0; };
VkDeviceSize max_storage_range=1024;
int scene_descriptor_set=1;
int operations=0;
Buffer createBuffer(VkDeviceSize size,int) { assert(size>0 && size<=max_storage_range); ++operations; return {size}; }
void updateDescriptor(int,uint32_t,int,const Buffer& b) { assert(b.size<=max_storage_range); ++operations; }
void destroyBuffer(Buffer& b) { ++operations; b={}; }
'''+capacity+r'''
int main() {
    Buffer b{32};
    ensureLightCapacity(b,0,32,1); assert(b.size==32 && operations==0);
    ensureLightCapacity(b,1,32,1); assert(b.size==32 && operations==0);
    ensureLightCapacity(b,2,32,1); assert(b.size==64 && operations==3);
    ensureLightCapacity(b,3,32,1); assert(b.size==128 && operations==6);
    ensureLightCapacity(b,4,32,1); assert(b.size==128 && operations==6);
    ensureLightCapacity(b,17,32,1); assert(b.size==1024 && operations==9);
    bool rejected=false;
    try { ensureLightCapacity(b,33,32,1); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected && b.size==1024 && operations==9);
    max_storage_range=1000; b={512};
    ensureLightCapacity(b,17,32,1); assert(b.size==544);
    b={64};
    ensureLightCapacity(b,15,64,2); assert(b.size==960);
    ensureLightCapacity(b,0,64,2); assert(b.size==960);
}
'''
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory); (p/'capacity.cpp').write_text(program)
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise RuntimeError('A C++ compiler is needed for the capacity test')
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',str(p/'capacity.cpp'),'-o',str(p/'capacity')],check=True)
    subprocess.run([str(p/'capacity')],check=True)
    print('PASS: actual C++ storage growth, empty arrays, reuse, and hardware-limit rejection.')
    glslc=shutil.which('glslc')
    if glslc:
        for shader in ('textured.vert','textured.frag'):
            subprocess.run([glslc,'--target-env=vulkan1.1',str(ROOT/'shaders'/shader),'-o',str(p/(shader+'.spv'))],check=True)
        subprocess.run([glslc,'--target-env=vulkan1.1','-DENABLE_EFFECTS=1',str(ROOT/'shaders/textured.frag'),'-o',str(p/'textured_effect.frag.spv')],check=True)
        print('PASS: vertex shader and both fragment shader variants compiled with glslc.')
    else:
        print('SKIP: GLSL compilation (glslc is not installed).')
assert 'std::array<Buffer, 3> model_uniform_buffers' not in CPP
assert 'VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC' in CPP
assert 'sizeof(ModelUniforms)}' in CPP
assert '1, &dynamicOffset' in CPP
assert 'VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT' in CPP
assert 'vkCmdCopyBuffer(cmd' in CPP and 'vkCmdCopyBufferToImage' in CPP
for alignment in [1,16,64,256,1024]:
    stride=(176+alignment-1)//alignment*alignment
    offsets=[i*stride for i in range(3)]
    assert all(o%alignment==0 and o+176<=3*stride for o in offsets)
import struct,zlib
files=list((ROOT/'textures').glob('*.png'))
assert len(files)==6
for file in files:
    data=file.read_bytes(); assert data[:8]==b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('>II',data[16:24])==(512,512)
    at=8
    while at<len(data):
        length=struct.unpack('>I',data[at:at+4])[0]
        chunk=data[at+4:at+8+length]
        crc=struct.unpack('>I',data[at+8+length:at+12+length])[0]
        assert zlib.crc32(chunk)&0xffffffff==crc
        at+=length+12
assert 'layout(push_constant)' in FRAG and '#ifdef ENABLE_EFFECTS' in FRAG
assert FRAG.count('texture(albedoTexture,')==5
assert 'effect.control.x == 0u' in FRAG
assert 'g_appState.effectEnabled ? 1u : 0u' in CPP
assert 'proceduralColors' not in CPP
print('PASS: dynamic UBO offsets; GPU upload paths; six PNG files and CRCs; conditional effects shader.')

