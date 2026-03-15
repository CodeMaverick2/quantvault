import winston from "winston";
import Transport from "winston-transport";

/**
 * Raw stderr transport — writes every log line directly via process.stderr.write.
 * Works regardless of TTY, nohup, pipe, or Claude tool environment.
 */
class RawStderrTransport extends Transport {
  log(info: any, callback: () => void) {
    const ts = new Date().toISOString().replace("T", " ").slice(0, 19);
    const level = (info.level ?? "info").toUpperCase();
    const meta = Object.keys(info).filter(k => !["level","message","timestamp"].includes(k)).length > 0
      ? " " + JSON.stringify(Object.fromEntries(Object.entries(info).filter(([k]) => !["level","message","timestamp"].includes(k))))
      : "";
    process.stderr.write(`${ts} ${level} [keeper] ${info.message}${meta}\n`);
    callback();
  }
}

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || "info",
  transports: [
    new RawStderrTransport(),
    new winston.transports.File({
      filename: "logs/keeper.log",
      maxsize: 10_000_000,
      maxFiles: 5,
      format: winston.format.combine(
        winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
        winston.format.errors({ stack: true }),
        winston.format.printf(({ timestamp, level, message, ...rest }) => {
          const meta = Object.keys(rest).length > 0 ? ` ${JSON.stringify(rest)}` : "";
          return `${timestamp} ${level.toUpperCase()} [keeper] ${message}${meta}`;
        })
      ),
    }),
  ],
});

export default logger;
