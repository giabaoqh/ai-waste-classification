export function toImagePoint(clientX, clientY, rect, width, height) {
  if (!(rect.width > 0 && rect.height > 0 && width > 0 && height > 0)) throw new Error('Invalid image geometry');
  return [Math.max(0, Math.min(width, (clientX - rect.left) / rect.width * width)),
    Math.max(0, Math.min(height, (clientY - rect.top) / rect.height * height))];
}

export function boxFromPoints(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}
