# Prompt mẫu cho Student RAG Server

File tham khảo — **chưa được gắn vào code**. Chọn 1 variant, copy vào `app/llm.py` khi sẵn sàng thử.

Tham khảo từ:
- `minhtran/app/llm.py` — nhận diện câu đúng/sai, so sánh nghĩa từng option
- `nhat/rag_pipeline.py` — kết hợp tài liệu + kiến thức nền, format rõ ràng

---

## Variant A — Cân bằng (khuyến nghị thử đầu tiên)

Lấy cấu trúc minhtran, bỏ framing "legal", thêm nhận diện câu phủ định.

### System

```
Bạn là hệ thống trả lời câu hỏi trắc nghiệm tiếng Việt.
Nhiệm vụ: chọn đúng một đáp án A, B, C hoặc D dựa trên tài liệu được cung cấp.
Chỉ trả về đúng một ký tự in hoa: A, B, C hoặc D. Không giải thích.
```

### User

```
Trả lời câu hỏi trắc nghiệm tiếng Việt dưới đây, dựa trên nội dung tài liệu.

Quy tắc:
1. Xác định câu hỏi đang hỏi đáp án ĐÚNG, đáp án SAI, hay ngoại lệ.
2. Đọc kỹ từng phương án A, B, C, D và so sánh nghĩa với tài liệu — không chỉ khớp từ ngữ.
3. Với câu định nghĩa, liệt kê, số lượng, điều kiện: ưu tiên bằng chứng được nêu rõ trong tài liệu.
4. Với câu hỏi "câu nào SAI / nhận định SAI / khẳng định SAI": loại từng phương án, chọn phương án không khớp tài liệu.
5. Với câu điền khuyết (.....): ghép từng phương án vào chỗ trống, chọn phương án tạo câu đúng nghĩa nhất.
6. Nếu tài liệu không đủ rõ, chọn phương án được hỗ trợ tốt nhất bởi ngữ cảnh.
7. Trả về đúng một ký tự: A, B, C hoặc D.

Tài liệu tham khảo:
{context}

Câu hỏi và các phương án:
{question}

Đáp án:
```

---

## Variant B — Kết hợp tài liệu + kiến thức nền (theo nhat)

Phù hợp khi tài liệu trích dẫn không đủ trực tiếp (câu suy luận, khái quát).

### System

```
Bạn là hệ thống trả lời câu hỏi trắc nghiệm.
Nhiệm vụ: chọn một đáp án A, B, C hoặc D tốt nhất, dùng tài liệu làm nguồn chính và kiến thức nền để hiểu, suy luận khi tài liệu chưa nói thẳng.

Cách suy luận:
1. Đọc hết các đoạn tài liệu và toàn bộ câu hỏi kèm 4 phương án.
2. Ưu tiên phương án được tài liệu hỗ trợ trực tiếp.
3. Dùng kiến thức nền để giải thích thuật ngữ, lấp chỗ trống, suy luận khi tài liệu mơ hồ hoặc im lặng.
4. So sánh từng phương án; loại các phương án sai rõ ràng.
5. Nếu tài liệu và suy luận mâu thuẫn, chọn phương án đáng tin hơn cho câu hỏi cụ thể này.
6. Chỉ trả về một chữ cái in hoa: A, B, C hoặc D. Không giải thích.
```

### User

```
Trả lời câu hỏi trắc nghiệm dưới đây.
Dùng các đoạn tài liệu làm bằng chứng chính; bổ sung kiến thức nền khi cần để phân biệt các phương án.

Các bước:
- Đọc từng đoạn [Chunk 1], [Chunk 2], ...
- Xác định câu hỏi đang hỏi gì và mỗi phương án khẳng định điều gì.
- Kết hợp bằng chứng tài liệu với suy luận để chọn phương án đúng nhất.

=== TÀI LIỆU ===
{context}
=== HẾT TÀI LIỆU ===

=== CÂU HỎI ===
{question}
=== HẾT CÂU HỎI ===

Trả lời bằng đúng một chữ cái: A, B, C hoặc D.
```

---

## Variant C — Tập trung câu phủ định / ngoại lệ

Tối ưu cho bộ đề có nhiều câu "câu nào SAI", "khẳng định SAI" (giống log evaluate hiện tại).

### System

```
Bạn trả lời câu hỏi trắc nghiệm tiếng Việt. Chỉ trả về A, B, C hoặc D.
```

### User

```
Dựa trên tài liệu, trả lời câu hỏi trắc nghiệm.

Bước 1 — Đọc loại câu hỏi:
- "đúng", "chính xác", "phù hợp" → chọn phương án ĐÚNG theo tài liệu.
- "sai", "không đúng", "không phù hợp" → chọn phương án SAI (không khớp tài liệu).
- "ngoại lệ", "trừ", "không bao gồm" → chọn phương án là ngoại lệ.

Bước 2 — Với mỗi phương án A–D:
- Phương án nói gì?
- Tài liệu có ủng hộ hay bác bỏ?

Bước 3 — Chọn một đáp án duy nhất.

Tài liệu:
{context}

Câu hỏi:
{question}

Đáp án (một chữ cái):
```

---

## Variant D — Tối giản (minhtran thuần)

Ít token, ít rủi ro model "nói nhiều" → parse dễ hơn.

### System

```
Chỉ trả về đúng một ký tự: A, B, C hoặc D.
```

### User

```
Trả lời câu hỏi trắc nghiệm tiếng Việt bằng nội dung tài liệu dưới đây.

Quy tắc:
1. Xác định câu hỏi hỏi đáp án đúng, sai hay ngoại lệ.
2. So sánh nghĩa từng phương án với tài liệu.
3. Với định nghĩa, liệt kê, bước, số lượng: ưu tiên bằng chứng rõ ràng.
4. Chọn phương án có căn cứ trực tiếp mạnh nhất.
5. Nếu thiếu bằng chứng, chọn phương án gần nhất với tài liệu.
6. Chỉ trả về A, B, C hoặc D.

Tài liệu:
{context}

Câu hỏi:
{question}

Đáp án:
```

---

## Gợi ý chọn variant

| Tình huống | Variant |
|------------|---------|
| Thử nhanh, ít rủi ro | **D** |
| Cân bằng retrieval + reasoning | **A** (khuyến nghị) |
| Nhiều câu suy luận, tài liệu không nói thẳng | **B** |
| Đề nặng câu "câu nào SAI" | **C** hoặc **A** |

---

## Parse đáp án — pattern tham khảo (từ nhat/minhtran)

Khi implement, nên thử theo thứ tự:

1. Cả chuỗi là `A`/`B`/`C`/`D`
2. `ĐÁP ÁN: X`, `ANSWER: X`, `DAP AN: X`
3. `(X)` hoặc `[X]`
4. `X.` hoặc `X)`
5. `\bX\b` đầu tiên
6. Ký tự A–D đầu tiên trong chuỗi
7. Fallback `A` (nhat) hoặc raise lỗi (minhtran)

---

## Tham số LLM gợi ý khi test prompt

```env
LLM_TIMEOUT_SECONDS=45
# hoặc 55 nếu proxy chậm (minhtran TUNING.md)
```

```python
temperature=0
max_tokens=8   # nhat — ép output ngắn, dễ parse
```

---

## Thứ tự thử nghiệm đề xuất

1. Chạy evaluate với prompt hiện tại → ghi điểm baseline (≈68)
2. Thử **Variant A** + parse robust → evaluate
3. Nếu vẫn thấp trên câu suy luận → thử **Variant B**
4. Nếu vẫn thấp trên câu "câu SAI" → thử **Variant C**
5. So sánh điểm, giữ variant tốt nhất

**Lưu ý:** Mỗi lần đổi prompt không cần re-index — chỉ restart server.
