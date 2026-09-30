import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outDir = "../config_render";
const input = await FileBlob.load(`${outDir}/Gryphon_math_sheet.xlsx`);
const workbook = await SpreadsheetFile.importXlsx(input);

console.log((await workbook.inspect({ kind: "sheet", include: "name" })).ndjson);

for (const sheetName of [
  "Summary",
  "RTP Build",
  "Odds",
  "Base Arrays",
  "Feature Arrays",
  "Game Rules",
  "Game Flow",
]) {
  const png = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.replace(/ /g, "_");
  await fs.writeFile(
    `${outDir}/render_${safeName}.png`,
    new Uint8Array(await png.arrayBuffer()),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "saved file formula error scan",
});
console.log(errors.ndjson);
