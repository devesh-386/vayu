// Adapted from Animmaster: Background Animations/3 (WebGL particle field).
// Changes: particles stand in for airborne particulate matter, so their count and tint are
// driven by the live AQI; rendering pauses off-screen; reduced motion draws one still frame.
import { useEffect, useRef } from 'react'

const VERT = `
attribute vec3 position;
attribute vec4 random;
attribute vec3 color;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float uTime;
uniform float uSpread;
uniform float uBaseSize;
uniform float uSizeRandomness;
varying vec4 vRandom;
varying vec3 vColor;
void main() {
  vRandom = random;
  vColor = color;
  vec3 pos = position * uSpread;
  pos.z *= 2.5;
  vec4 mPos = modelMatrix * vec4(pos, 1.0);
  float t = uTime;
  mPos.x += sin(t * random.z + 6.28 * random.w) * mix(0.1, 1.5, random.x);
  mPos.y += sin(t * random.y + 6.28 * random.x) * mix(0.1, 1.5, random.w);
  mPos.z += sin(t * random.w + 6.28 * random.y) * mix(0.1, 1.5, random.z);
  vec4 mvPos = viewMatrix * mPos;
  gl_PointSize = (uBaseSize * (1.0 + uSizeRandomness * (random.x - 0.5))) / length(mvPos.xyz);
  gl_Position = projectionMatrix * mvPos;
}`

const FRAG = `
precision highp float;
uniform float uTime;
uniform float uOpacity;
varying vec4 vRandom;
varying vec3 vColor;
void main() {
  vec2 uv = gl_PointCoord.xy;
  float d = length(uv - vec2(0.5));
  float circle = smoothstep(0.5, 0.15, d) * uOpacity * mix(0.35, 1.0, vRandom.z);
  gl_FragColor = vec4(vColor + 0.08 * sin(uv.yxx + uTime + vRandom.y * 6.28), circle);
}`

const hexToRgb = (hex: string) => {
  const n = parseInt(hex.replace('#', ''), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

// Minimal mat4 helpers (column-major), same maths as the source component.
const identity = (o: Float32Array) => { o.fill(0); o[0] = o[5] = o[10] = o[15] = 1; return o }
const perspective = (o: Float32Array, fovy: number, aspect: number, near: number, far: number) => {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far)
  o.fill(0); o[0] = f / aspect; o[5] = f; o[10] = (far + near) * nf; o[11] = -1; o[14] = 2 * far * near * nf
  return o
}
const multiply = (a: Float32Array, b: Float32Array) => {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3]
  return o
}
/** model = T(tx, ty) * Rx * Ry * Rz, the same order as the source component. */
const rotation = (o: Float32Array, x: number, y: number, z: number, tx: number, ty: number) => {
  const t = identity(new Float32Array(16)); t[12] = tx; t[13] = ty
  const rx = identity(new Float32Array(16)); rx[5] = Math.cos(x); rx[6] = Math.sin(x); rx[9] = -Math.sin(x); rx[10] = Math.cos(x)
  const ry = identity(new Float32Array(16)); ry[0] = Math.cos(y); ry[2] = -Math.sin(y); ry[8] = Math.sin(y); ry[10] = Math.cos(y)
  const rz = identity(new Float32Array(16)); rz[0] = Math.cos(z); rz[1] = Math.sin(z); rz[4] = -Math.sin(z); rz[5] = Math.cos(z)
  o.set(multiply(multiply(multiply(t, rx), ry), rz))
  return o
}

type Props = {
  /** AQI 0-500: sets how dense the haze is. */
  aqi: number
  colors: string[]
  className?: string
}

export function ParticleField({ aqi, colors, className }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const colorKey = colors.join(',')

  useEffect(() => {
    const container = ref.current
    if (!container) return
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const count = Math.round(220 + Math.min(Math.max(aqi, 0), 500) * 2.6)
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    const canvas = document.createElement('canvas')
    canvas.style.cssText = 'width:100%;height:100%;display:block'
    canvas.setAttribute('aria-hidden', 'true')
    container.appendChild(canvas)
    const gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: false })
    if (!gl) return () => { canvas.remove() }

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!
      gl.shaderSource(s, src); gl.compileShader(s)
      return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null
    }
    const vs = compile(gl.VERTEX_SHADER, VERT), fs = compile(gl.FRAGMENT_SHADER, FRAG)
    if (!vs || !fs) return () => { canvas.remove() }
    const program = gl.createProgram()!
    gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program)
    gl.useProgram(program)
    gl.clearColor(0, 0, 0, 0)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    const positions = new Float32Array(count * 3)
    const randoms = new Float32Array(count * 4)
    const cols = new Float32Array(count * 3)
    const palette = colors.map(hexToRgb)
    for (let i = 0; i < count; i++) {
      let x, y, z, len
      do {
        x = Math.random() * 2 - 1; y = Math.random() * 2 - 1; z = Math.random() * 2 - 1
        len = x * x + y * y + z * z
      } while (len > 1 || len === 0)
      const r = Math.cbrt(Math.random())
      positions.set([x * r, y * r, z * r], i * 3)
      randoms.set([Math.random(), Math.random(), Math.random(), Math.random()], i * 4)
      cols.set(palette[Math.floor(Math.random() * palette.length)], i * 3)
    }
    const attr = (name: string, data: Float32Array, size: number) => {
      const loc = gl.getAttribLocation(program, name)
      const buf = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, buf)
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
      gl.enableVertexAttribArray(loc)
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0)
      return buf
    }
    const bufs = [attr('position', positions, 3), attr('random', randoms, 4), attr('color', cols, 3)]
    const u = (n: string) => gl.getUniformLocation(program, n)
    gl.uniform1f(u('uSpread'), 13)
    gl.uniform1f(u('uBaseSize'), 200 * dpr)
    gl.uniform1f(u('uSizeRandomness'), 1)
    // Heavier air reads as a denser, more opaque haze.
    gl.uniform1f(u('uOpacity'), 0.55 + Math.min(aqi, 400) / 400 * 0.4)
    const view = identity(new Float32Array(16)); view[14] = -20
    gl.uniformMatrix4fv(u('viewMatrix'), false, view)
    const proj = new Float32Array(16), model = new Float32Array(16)

    const resize = () => {
      const w = Math.max(1, Math.floor(container.clientWidth * dpr))
      const h = Math.max(1, Math.floor(container.clientHeight * dpr))
      canvas.width = w; canvas.height = h
      gl.viewport(0, 0, w, h)
      perspective(proj, (45 * Math.PI) / 180, w / h, 0.1, 1000)
      gl.uniformMatrix4fv(u('projectionMatrix'), false, proj)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    resize()

    const mouse = { x: 0, y: 0, tx: 0, ty: 0 }
    const onMove = (e: PointerEvent) => {
      const r = container.getBoundingClientRect()
      mouse.tx = ((e.clientX - r.left) / r.width) * 2 - 1
      mouse.ty = -(((e.clientY - r.top) / r.height) * 2 - 1)
    }
    if (!reduce) window.addEventListener('pointermove', onMove, { passive: true })

    const speed = 0.1
    let raf = 0, last = performance.now(), elapsed = 0, rotZ = 0, visible = true
    const draw = () => {
      // Ease towards the pointer so the haze drifts rather than snaps.
      mouse.x += (mouse.tx - mouse.x) * 0.04
      mouse.y += (mouse.ty - mouse.y) * 0.04
      gl.uniform1f(u('uTime'), elapsed * 1e-3)
      rotation(model, Math.sin(elapsed * 2e-4) * 0.1, Math.cos(elapsed * 5e-4) * 0.15, rotZ, -mouse.x * 0.6, -mouse.y * 0.6)
      gl.uniformMatrix4fv(u('modelMatrix'), false, model)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.POINTS, 0, count)
    }
    const tick = (t: number) => {
      raf = requestAnimationFrame(tick)
      const dt = Math.min(t - last, 64); last = t
      elapsed += dt * speed
      rotZ += 0.004 * speed * (dt / 16)
      draw()
    }
    const start = () => { if (!raf && visible && !reduce) { last = performance.now(); raf = requestAnimationFrame(tick) } }
    const stop = () => { cancelAnimationFrame(raf); raf = 0 }
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) start(); else stop() })
    io.observe(container)
    if (reduce) { elapsed = 4000; draw() } else start()

    return () => {
      stop(); io.disconnect(); ro.disconnect()
      window.removeEventListener('pointermove', onMove)
      bufs.forEach(b => gl.deleteBuffer(b))
      gl.deleteProgram(program)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
      canvas.remove()
    }
  }, [aqi, colorKey]) // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={ref} className={className} />
}
