/* Copy Yahoo's draft picks in a form the draft board can read.
 *
 * Paste into the Console while the draft room's "Picks" view is open. It
 * copies one line per pick to your clipboard; paste that into the board's
 * "Paste picks from your draft room" box.
 *
 * It finds the list structurally -- the element with the most direct children
 * each holding exactly one "X. Surname" -- rather than by CSS class. Yahoo's
 * classes are generated (`W(150px)`, `Fxg(1)`) and change whenever they
 * rebuild, so a class selector would break silently and probably mid-draft.
 * The structure is more durable, though nothing here is guaranteed.
 *
 * The one thing that must not be done naively is reading textContent: it
 * concatenates adjacent elements with no separator, so a row arrives as
 * "1PopCornN. MacKinnonCCOL" and nothing matches. Leaf texts are joined with
 * spaces instead, which yields "1 PopCorn N. MacKinnon C COL".
 */
(() => {
  const pat = /\b[A-Z]\.\s?[A-Z][a-zA-Z'’-]{2,}/;

  const leaves = [...document.querySelectorAll("*")]
    .filter(el => !el.children.length && pat.test(el.textContent || ""));
  if (!leaves.length) {
    console.log("No abbreviated names on the page. Is the Picks view open?");
    return;
  }

  // The list is the ancestor with the most direct children bearing a name.
  const kids = new Map();
  for (const leaf of leaves) {
    let child = leaf;
    for (let p = leaf.parentElement; p; child = p, p = p.parentElement) {
      if (!kids.has(p)) kids.set(p, new Set());
      kids.get(p).add(child);
    }
  }
  const best = [...kids.entries()]
    .map(([el, set]) => ({ el, n: set.size }))
    .filter(x => x.n >= 5)
    .sort((a, b) => b.n - a.n)[0];
  if (!best) { console.log("Found names but no repeating list around them."); return; }

  // Spaces between leaves, or the fields run together and nothing matches.
  const rowText = row => [...row.querySelectorAll("*")]
    .filter(e => !e.children.length && e.textContent.trim())
    .map(e => e.textContent.trim())
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

  const rows = [...best.el.children].map(rowText)
    .filter(t => pat.test(t));   // drops "Sahara joined" and other chat noise

  if (!rows.length) { console.log("List found but no rows looked like picks."); return; }

  copy(rows.join("\n"));
  console.log(`copied ${rows.length} picks`);
  console.log(rows.slice(0, 5));
})();
