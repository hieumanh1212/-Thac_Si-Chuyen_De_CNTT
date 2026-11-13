import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Hộp xác nhận dùng chung, thay cho window.confirm của trình duyệt.
 *
 * Cách dùng:
 *   const [confirm, dialog] = useConfirm();
 *   if (await confirm({ title: "Xóa?", message: "...", confirmLabel: "Xóa", danger: true })) { ... }
 *   return <>{dialog} ...</>;
 */
export function useConfirm() {
  const [options, setOptions] = useState(null);
  const resolver = useRef(null);

  const confirm = useCallback((opts) => new Promise((resolve) => {
    resolver.current = resolve;
    setOptions(opts);
  }), []);

  const close = useCallback((result) => {
    resolver.current?.(result);
    resolver.current = null;
    setOptions(null);
  }, []);

  const dialog = options ? <ConfirmDialog {...options} onClose={close} /> : null;
  return [confirm, dialog];
}

export default function ConfirmDialog({
  title, message, confirmLabel = "Đồng ý", cancelLabel = "Hủy", danger = false, onClose,
}) {
  const cancelRef = useRef(null);

  useEffect(() => {
    cancelRef.current?.focus();                    // mặc định focus nút Hủy cho an toàn
    const onKey = (e) => { if (e.key === "Escape") onClose(false); };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";       // khóa cuộn trang phía sau
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(false); }}>
      <div className="modal" role="alertdialog" aria-modal="true" aria-labelledby="modal-title">
        <div className="modal-head">
          <span className={danger ? "modal-icon danger" : "modal-icon"} aria-hidden="true">{danger ? "!" : "?"}</span>
          <h2 id="modal-title">{title}</h2>
        </div>
        {message && <div className="modal-body">{message}</div>}
        <div className="modal-actions">
          <button ref={cancelRef} className="btn" onClick={() => onClose(false)}>{cancelLabel}</button>
          <button className={danger ? "btn danger-solid" : "btn primary"} onClick={() => onClose(true)}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
