"""
Design System for professional Next.js frontends.
Generates project scaffolding with modern stack:
- Next.js 14 (App Router)
- Tailwind CSS
- Framer Motion
- Lucide Icons
- Shadcn/ui components
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """Configuration for a Next.js project."""

    name: str
    description: str = ""
    author: str = ""
    version: str = "0.1.0"

    # Features to include
    use_typescript: bool = True
    use_tailwind: bool = True
    use_framer_motion: bool = True
    use_lucide: bool = True
    use_shadcn: bool = True

    # Additional settings
    src_dir: bool = True  # Use src/ directory
    app_router: bool = True  # Use App Router (vs Pages)


class DesignSystem:
    """
    Next.js 14 project scaffolding with professional stack.

    Generates a complete project structure ready for deployment to Vercel.
    """

    # Stack versions
    STACK = {
        "next": "14.1.0",
        "react": "18.2.0",
        "tailwindcss": "3.4.1",
        "framer-motion": "11.0.3",
        "lucide-react": "0.321.0",
    }

    def create_project(self, config: ProjectConfig) -> Dict[str, str]:
        """
        Generate all files for a Next.js project.

        Args:
            config: Project configuration

        Returns:
            Dict of {filepath: content}
        """
        files = {}

        # Package configuration
        files["package.json"] = self._generate_package_json(config)

        # TypeScript config
        if config.use_typescript:
            files["tsconfig.json"] = self._generate_tsconfig()

        # Tailwind configuration
        if config.use_tailwind:
            files["tailwind.config.ts"] = self._generate_tailwind_config(config)
            files["postcss.config.js"] = self._generate_postcss_config()

        # Next.js config
        files["next.config.js"] = self._generate_next_config()

        # Source files
        src_prefix = "src/" if config.src_dir else ""

        # App Router structure
        if config.app_router:
            files[f"{src_prefix}app/layout.tsx"] = self._generate_layout(config)
            files[f"{src_prefix}app/page.tsx"] = self._generate_homepage(config)
            files[f"{src_prefix}app/globals.css"] = self._generate_globals_css(config)

        # Utility files
        files[f"{src_prefix}lib/utils.ts"] = self._generate_utils()

        # Component directories
        files[f"{src_prefix}components/.gitkeep"] = ""

        # Environment example
        files[".env.example"] = self._generate_env_example()

        # Git files
        files[".gitignore"] = self._generate_gitignore()

        # README
        files["README.md"] = self._generate_readme(config)

        return files

    def _generate_package_json(self, config: ProjectConfig) -> str:
        """Generate package.json content."""
        ext = "tsx" if config.use_typescript else "jsx"

        dependencies = {
            "next": f"^{self.STACK['next']}",
            "react": f"^{self.STACK['react']}",
            "react-dom": f"^{self.STACK['react']}",
        }

        dev_dependencies = {
            "eslint": "^8.56.0",
            "eslint-config-next": f"^{self.STACK['next']}",
        }

        if config.use_typescript:
            dev_dependencies.update({
                "typescript": "^5.3.3",
                "@types/node": "^20.11.5",
                "@types/react": "^18.2.48",
                "@types/react-dom": "^18.2.18",
            })

        if config.use_tailwind:
            dev_dependencies.update({
                "tailwindcss": f"^{self.STACK['tailwindcss']}",
                "postcss": "^8.4.33",
                "autoprefixer": "^10.4.17",
            })

        if config.use_framer_motion:
            dependencies["framer-motion"] = f"^{self.STACK['framer-motion']}"

        if config.use_lucide:
            dependencies["lucide-react"] = f"^{self.STACK['lucide-react']}"

        if config.use_shadcn:
            dependencies.update({
                "class-variance-authority": "^0.7.0",
                "clsx": "^2.1.0",
                "tailwind-merge": "^2.2.1",
                "@radix-ui/react-slot": "^1.0.2",
            })

        package = {
            "name": config.name.lower().replace(" ", "-"),
            "version": config.version,
            "description": config.description,
            "private": True,
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint",
            },
            "dependencies": dict(sorted(dependencies.items())),
            "devDependencies": dict(sorted(dev_dependencies.items())),
        }

        import json
        return json.dumps(package, indent=2)

    def _generate_tsconfig(self) -> str:
        """Generate TypeScript configuration."""
        return """{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}"""

    def _generate_tailwind_config(self, config: ProjectConfig) -> str:
        """Generate Tailwind CSS configuration."""
        content_paths = [
            '"./src/pages/**/*.{js,ts,jsx,tsx,mdx}"',
            '"./src/components/**/*.{js,ts,jsx,tsx,mdx}"',
            '"./src/app/**/*.{js,ts,jsx,tsx,mdx}"',
        ] if config.src_dir else [
            '"./pages/**/*.{js,ts,jsx,tsx,mdx}"',
            '"./components/**/*.{js,ts,jsx,tsx,mdx}"',
            '"./app/**/*.{js,ts,jsx,tsx,mdx}"',
        ]

        content_paths_str = ',\n    '.join(content_paths)

        return f"""import type {{ Config }} from "tailwindcss"

const config: Config = {{
  content: [
    {content_paths_str},
  ],
  theme: {{
    extend: {{
      colors: {{
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {{
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        }},
        secondary: {{
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        }},
        destructive: {{
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        }},
        muted: {{
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        }},
        accent: {{
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        }},
        card: {{
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        }},
      }},
      borderRadius: {{
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      }},
      keyframes: {{
        "fade-in": {{
          "0%": {{ opacity: "0", transform: "translateY(10px)" }},
          "100%": {{ opacity: "1", transform: "translateY(0)" }},
        }},
        "fade-out": {{
          "0%": {{ opacity: "1", transform: "translateY(0)" }},
          "100%": {{ opacity: "0", transform: "translateY(10px)" }},
        }},
      }},
      animation: {{
        "fade-in": "fade-in 0.3s ease-out",
        "fade-out": "fade-out 0.3s ease-out",
      }},
    }},
  }},
  plugins: [],
}}

export default config"""

    def _generate_postcss_config(self) -> str:
        """Generate PostCSS configuration."""
        return """module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}"""

    def _generate_next_config(self) -> str:
        """Generate Next.js configuration."""
        return """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
}

module.exports = nextConfig"""

    def _generate_layout(self, config: ProjectConfig) -> str:
        """Generate root layout component."""
        imports = ['import type { Metadata } from "next"']
        imports.append('import { Inter } from "next/font/google"')
        imports.append('import "./globals.css"')

        return f"""{chr(10).join(imports)}

const inter = Inter({{ subsets: ["latin"] }})

export const metadata: Metadata = {{
  title: "{config.name}",
  description: "{config.description or 'A modern web application'}",
}}

export default function RootLayout({{
  children,
}}: {{
  children: React.ReactNode
}}) {{
  return (
    <html lang="en">
      <body className={{inter.className}}>{{children}}</body>
    </html>
  )
}}"""

    def _generate_homepage(self, config: ProjectConfig) -> str:
        """Generate homepage component."""
        icon_import = ""
        icon_usage = ""

        if config.use_lucide:
            icon_import = 'import { Rocket, Github, ArrowRight } from "lucide-react"'
            icon_usage = "<Rocket className=\"h-12 w-12 text-primary\" />"
        else:
            icon_usage = '<span className="text-4xl">🚀</span>'

        motion_wrapper_open = ""
        motion_wrapper_close = ""

        if config.use_framer_motion:
            motion_import = '"use client"\n\nimport { motion } from "framer-motion"'
            motion_wrapper_open = '''<motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >'''
            motion_wrapper_close = "</motion.div>"
        else:
            motion_import = ""

        return f'''{motion_import}
{icon_import}

export default function Home() {{
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-background">
      {motion_wrapper_open}
        <div className="text-center space-y-6">
          <div className="flex justify-center">
            {icon_usage}
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-6xl">
            {config.name}
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl">
            {config.description or "A modern web application built with Next.js, Tailwind CSS, and more."}
          </p>
          <div className="flex gap-4 justify-center">
            <a
              href="#"
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 transition-colors"
            >
              Get Started
            </a>
            <a
              href="#"
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              Learn More
            </a>
          </div>
        </div>
      {motion_wrapper_close}
    </main>
  )
}}'''

    def _generate_globals_css(self, config: ProjectConfig) -> str:
        """Generate global CSS with Tailwind and CSS variables."""
        return """@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}"""

    def _generate_utils(self) -> str:
        """Generate utility functions."""
        return '''import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}'''

    def _generate_env_example(self) -> str:
        """Generate environment variables example."""
        return """# Environment Variables
# Copy this file to .env.local and fill in the values

# API Keys
# NEXT_PUBLIC_API_URL=https://api.example.com

# Authentication
# NEXTAUTH_SECRET=your-secret-here
# NEXTAUTH_URL=http://localhost:3000
"""

    def _generate_gitignore(self) -> str:
        """Generate .gitignore file."""
        return """# Dependencies
/node_modules
/.pnp
.pnp.js

# Testing
/coverage

# Next.js
/.next/
/out/

# Production
/build

# Misc
.DS_Store
*.pem

# Debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Local env files
.env*.local

# Vercel
.vercel

# TypeScript
*.tsbuildinfo
next-env.d.ts
"""

    def _generate_readme(self, config: ProjectConfig) -> str:
        """Generate README.md file."""
        return f"""# {config.name}

{config.description or "A modern web application."}

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **Icons**: Lucide React
- **Components**: Shadcn/ui patterns

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Copy environment variables:
   ```bash
   cp .env.example .env.local
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000)

## Deployment

Deploy to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

## Project Structure

```
{config.name}/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   └── lib/
│       └── utils.ts
├── public/
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

---

Generated by AI Orchestrator
"""


# Convenience function
def create_nextjs_project(
    name: str,
    description: str = "",
    **kwargs,
) -> Dict[str, str]:
    """
    Create a Next.js project with default professional settings.

    Args:
        name: Project name
        description: Project description
        **kwargs: Additional ProjectConfig options

    Returns:
        Dict of {filepath: content}
    """
    config = ProjectConfig(name=name, description=description, **kwargs)
    system = DesignSystem()
    return system.create_project(config)
