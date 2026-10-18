export function readyMessageAttachments(items) {
  return items.filter((item) => item.status === "Ready" && item.serverAttachmentId);
}

export function consumeMessageAttachments(items, sentItems) {
  const sentIds = new Set(sentItems.map((item) => item.id));
  return items.filter((item) => !sentIds.has(item.id));
}
