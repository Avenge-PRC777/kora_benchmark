const MCU_HERO_NAMES = [
  "IronMan",
  "CaptainAmerica",
  "Thor",
  "HulkBanner",
  "BlackWidow",
  "Hawkeye",
  "NickFury",
  "SpiderMan",
  "DoctorStrange",
  "BlackPanther",
  "CaptainMarvel",
  "AntMan",
  "Wasp",
  "ScarletWitch",
  "Vision",
  "FalconWilson",
  "WinterSoldier",
  "WarMachine",
  "Gamora",
  "StarLord",
  "Drax",
  "Rocket",
  "Groot",
  "Mantis",
  "Nebula",
  "Loki",
  "Valkyrie",
  "ShangChi",
  "MsMarvel",
  "MoonKnight",
  "Deadpool",
  "Wolverine",
  "Daredevil",
  "LukeCage",
  "IronFist",
  "JessicaJones",
  "NovaRider",
];

function randomHeroName(): string {
  const name = MCU_HERO_NAMES[Math.floor(Math.random() * MCU_HERO_NAMES.length)];
  if (!name) throw new Error("MCU_HERO_NAMES is empty.");
  return name;
}

// Formats the given instant in the America/Los_Angeles time zone as e.g.
// "august11_5:40:23PM" (lowercase month, no zero-padded day/hour, uppercase
// meridiem glued to the seconds).
function formatPacificTimestamp(date: Date): string {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Los_Angeles",
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    })
      .formatToParts(date)
      .map(part => [part.type, part.value])
  );

  const month = parts.month!.toLowerCase();
  const meridiem = parts.dayPeriod!.toUpperCase();
  return `${month}${parts.day}_${parts.hour}:${parts.minute}:${parts.second}${meridiem}`;
}

/** e.g. "IronMan_august11_5:40:23PM" — a fresh name for each auto-created run folder. */
export function generateRunFolderName(now: Date = new Date()): string {
  return `${randomHeroName()}_${formatPacificTimestamp(now)}`;
}
