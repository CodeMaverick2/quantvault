import winston from "winston";

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || "info",
  format: winston.format.combine(
    winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
    winston.format.errors({ stack: true }),
    winston.format.printf(({ timestamp, level, message, ...rest }) => {
      const meta = Object.keys(rest).length > 0 ? ` ${JSON.stringify(rest)}` : "";
      return `${timestamp} ${level.toUpperCase()} [keeper] ${message}${meta}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: "logs/keeper.log", maxsize: 10_000_000, maxFiles: 5 }),
  ],
});

export default logger;
