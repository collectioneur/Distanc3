import { useEffect, useMemo, useRef, useState } from "react";
import { useSceneStore } from "../../store/sceneStore";
import { generateSdfCode, type CodeLang } from "../../utils/generateSdfCode";

const LANGS: { id: CodeLang; label: string; shiki: string }[] = [
  { id: "typegpu", label: "TypeGPU", shiki: "typescript" },
  { id: "wgsl", label: "WGSL", shiki: "wgsl" },
  { id: "glsl", label: "GLSL", shiki: "glsl" },
];

// ── Lazy shiki highlighter (loaded on first Code tab open) ───────────────────

type Highlighter = { codeToHtml(code: string, opts: { lang: string; theme: string }): string };

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  highlighterPromise ??= (async () => {
    const [{ createHighlighterCore }, { createJavaScriptRegexEngine }, ts, wgsl, glsl, theme] =
      await Promise.all([
        import("shiki/core"),
        import("shiki/engine/javascript"),
        import("@shikijs/langs/typescript"),
        import("@shikijs/langs/wgsl"),
        import("@shikijs/langs/glsl"),
        import("@shikijs/themes/github-dark-default"),
      ]);
    return createHighlighterCore({
      langs: [ts.default, wgsl.default, glsl.default],
      themes: [theme.default],
      engine: createJavaScriptRegexEngine({ forgiving: true }),
    });
  })();
  return highlighterPromise;
}

export default function CodeView() {
  const root = useSceneStore((s) => s.root);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const [lang, setLang] = useState<CodeLang>("typegpu");
  const [copied, setCopied] = useState(false);
  const [html, setHtml] = useState<string | null>(null);

  const generated = useMemo(() => generateSdfCode(root, selectedItemId), [root, selectedItemId]);
  const code = generated[lang];

  useEffect(() => {
    let cancelled = false;
    const shikiLang = LANGS.find((l) => l.id === lang)!.shiki;
    getHighlighter().then(
      (hl) => {
        if (!cancelled)
          setHtml(hl.codeToHtml(code, { lang: shikiLang, theme: "github-dark-default" }));
      },
      () => {
        if (!cancelled) setHtml(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [code, lang]);

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(id);
  }, [copied]);

  // Drag-to-scroll for the language strip; a click after a real drag is ignored.
  const stripRef = useRef<HTMLDivElement>(null);
  const drag = useRef({ startX: 0, startScroll: 0, moved: false });

  function onStripPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    const strip = stripRef.current;
    if (!strip) return;
    drag.current = { startX: e.clientX, startScroll: strip.scrollLeft, moved: false };
  }

  function onStripPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const strip = stripRef.current;
    if (!strip || e.buttons !== 1) return;
    const dx = e.clientX - drag.current.startX;
    // Capture only once a real drag starts, so plain clicks reach the buttons.
    if (!drag.current.moved && Math.abs(dx) <= 4) return;
    if (!drag.current.moved) {
      drag.current.moved = true;
      strip.setPointerCapture(e.pointerId);
    }
    strip.scrollLeft = drag.current.startScroll - dx;
  }

  function selectLang(id: CodeLang) {
    if (drag.current.moved) return;
    setLang(id);
  }

  return (
    <div className="code-view">
      <div
        ref={stripRef}
        className="code-langs"
        onPointerDown={onStripPointerDown}
        onPointerMove={onStripPointerMove}
      >
        {LANGS.map((l) => (
          <button
            key={l.id}
            className={`code-lang${lang === l.id ? " code-lang--active" : ""}`}
            onClick={() => selectLang(l.id)}
          >
            {l.label}
          </button>
        ))}
      </div>
      <div className="code-body">
        <button
          className="code-copy-btn"
          onClick={() => navigator.clipboard.writeText(code).then(() => setCopied(true))}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        {html ? (
          <div className="code-pre" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <pre className="code-pre">{code}</pre>
        )}
      </div>
    </div>
  );
}
