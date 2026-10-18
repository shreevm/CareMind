import test from "node:test";
import assert from "node:assert/strict";
import { readyMessageAttachments, consumeMessageAttachments } from "../app/composer-attachments.mjs";

test("sending a report does not attach it to the next question", () => {
  const uploaded = { id: "one", status: "Ready", serverAttachmentId: "report" };
  const pending = { id: "two", status: "Uploading" };
  const library = [uploaded];
  const composer = [uploaded, pending];
  const sent = readyMessageAttachments(composer);
  assert.deepEqual(sent.map((item) => item.serverAttachmentId), ["report"]);
  const remaining = consumeMessageAttachments(composer, sent);
  assert.deepEqual(remaining, [pending]);
  assert.deepEqual(readyMessageAttachments(remaining), []);
  assert.deepEqual(library, [uploaded]);
  assert.deepEqual(composer, [uploaded, pending]);
});

test("a file added during a send remains available to the next message", () => {
  const first = { id: "one", status: "Ready", serverAttachmentId: "a" };
  const second = { id: "two", status: "Ready", serverAttachmentId: "b" };
  assert.deepEqual(consumeMessageAttachments([first, second], [first]), [second]);
});
