"""
Component Library for professional React frontends.
Provides pre-built, styled components using Shadcn/ui patterns.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ComponentCategory(Enum):
    """Categories of components."""

    PRIMITIVE = "primitive"  # Basic UI elements
    FORM = "form"  # Form controls
    LAYOUT = "layout"  # Page layout components
    FEEDBACK = "feedback"  # User feedback (toasts, modals)
    DATA = "data"  # Data display (tables, charts)
    NAVIGATION = "navigation"  # Navigation elements


@dataclass
class ComponentInfo:
    """Information about a component."""

    name: str
    category: ComponentCategory
    description: str
    props: List[str]
    has_variants: bool = False


class ComponentLibrary:
    """
    Pre-built React components for professional frontends.

    Components follow Shadcn/ui patterns with:
    - Tailwind CSS styling
    - Class Variance Authority for variants
    - Full TypeScript support
    - Accessible by default
    """

    # Component registry
    COMPONENTS: Dict[str, ComponentInfo] = {
        # Primitives
        "Button": ComponentInfo(
            name="Button",
            category=ComponentCategory.PRIMITIVE,
            description="Clickable button with variants",
            props=["variant", "size", "asChild", "disabled"],
            has_variants=True,
        ),
        "Card": ComponentInfo(
            name="Card",
            category=ComponentCategory.PRIMITIVE,
            description="Container card with header, content, footer",
            props=["className"],
        ),
        "Badge": ComponentInfo(
            name="Badge",
            category=ComponentCategory.PRIMITIVE,
            description="Small label badge",
            props=["variant"],
            has_variants=True,
        ),
        # Form components
        "Input": ComponentInfo(
            name="Input",
            category=ComponentCategory.FORM,
            description="Text input field",
            props=["type", "placeholder", "disabled"],
        ),
        "Select": ComponentInfo(
            name="Select",
            category=ComponentCategory.FORM,
            description="Dropdown select component",
            props=["options", "placeholder", "disabled"],
        ),
        "Textarea": ComponentInfo(
            name="Textarea",
            category=ComponentCategory.FORM,
            description="Multi-line text input",
            props=["placeholder", "rows", "disabled"],
        ),
        "Checkbox": ComponentInfo(
            name="Checkbox",
            category=ComponentCategory.FORM,
            description="Checkbox input",
            props=["checked", "onCheckedChange", "disabled"],
        ),
        # Layout components
        "Navbar": ComponentInfo(
            name="Navbar",
            category=ComponentCategory.LAYOUT,
            description="Top navigation bar",
            props=["logo", "links", "actions"],
        ),
        "Sidebar": ComponentInfo(
            name="Sidebar",
            category=ComponentCategory.LAYOUT,
            description="Collapsible side navigation",
            props=["items", "collapsed", "onCollapse"],
        ),
        "Footer": ComponentInfo(
            name="Footer",
            category=ComponentCategory.LAYOUT,
            description="Page footer with links",
            props=["links", "copyright"],
        ),
        "Hero": ComponentInfo(
            name="Hero",
            category=ComponentCategory.LAYOUT,
            description="Hero section with CTA",
            props=["title", "subtitle", "cta", "image"],
        ),
        # Feedback components
        "Modal": ComponentInfo(
            name="Modal",
            category=ComponentCategory.FEEDBACK,
            description="Dialog modal overlay",
            props=["open", "onOpenChange", "title", "description"],
        ),
        "Toast": ComponentInfo(
            name="Toast",
            category=ComponentCategory.FEEDBACK,
            description="Toast notification",
            props=["title", "description", "variant"],
            has_variants=True,
        ),
        "LoadingSpinner": ComponentInfo(
            name="LoadingSpinner",
            category=ComponentCategory.FEEDBACK,
            description="Loading indicator",
            props=["size"],
        ),
        # Data components
        "DataTable": ComponentInfo(
            name="DataTable",
            category=ComponentCategory.DATA,
            description="Sortable data table",
            props=["columns", "data", "pagination"],
        ),
        "Tabs": ComponentInfo(
            name="Tabs",
            category=ComponentCategory.DATA,
            description="Tabbed content panels",
            props=["tabs", "defaultValue"],
        ),
        # Navigation
        "Breadcrumb": ComponentInfo(
            name="Breadcrumb",
            category=ComponentCategory.NAVIGATION,
            description="Breadcrumb navigation",
            props=["items"],
        ),
        "Dropdown": ComponentInfo(
            name="Dropdown",
            category=ComponentCategory.NAVIGATION,
            description="Dropdown menu",
            props=["trigger", "items"],
        ),
    }

    # Component templates
    TEMPLATES = {
        "Button": '''import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }''',

        "Card": '''import * as React from "react"
import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border bg-card text-card-foreground shadow-sm",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }''',

        "Input": '''import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }''',

        "Badge": '''import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive: "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }''',

        "LoadingSpinner": '''import * as React from "react"
import { cn } from "@/lib/utils"

interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: "sm" | "md" | "lg"
}

const sizeClasses = {
  sm: "h-4 w-4",
  md: "h-8 w-8",
  lg: "h-12 w-12",
}

export function LoadingSpinner({
  className,
  size = "md",
  ...props
}: LoadingSpinnerProps) {
  return (
    <div
      className={cn(
        "animate-spin rounded-full border-2 border-current border-t-transparent text-primary",
        sizeClasses[size],
        className
      )}
      {...props}
    >
      <span className="sr-only">Loading...</span>
    </div>
  )
}''',

        "Navbar": '''import * as React from "react"
import Link from "next/link"
import { cn } from "@/lib/utils"

interface NavbarProps {
  logo?: React.ReactNode
  links?: Array<{ href: string; label: string }>
  actions?: React.ReactNode
  className?: string
}

export function Navbar({ logo, links = [], actions, className }: NavbarProps) {
  return (
    <header
      className={cn(
        "sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
        className
      )}
    >
      <div className="container flex h-14 items-center">
        <div className="mr-4 flex">
          {logo && <Link href="/" className="mr-6 flex items-center space-x-2">{logo}</Link>}
          <nav className="flex items-center space-x-6 text-sm font-medium">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="transition-colors hover:text-foreground/80 text-foreground/60"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-end space-x-2">
          {actions}
        </div>
      </div>
    </header>
  )
}''',

        "Hero": '''import * as React from "react"
import { cn } from "@/lib/utils"

interface HeroProps {
  title: string
  subtitle?: string
  cta?: React.ReactNode
  image?: string
  className?: string
}

export function Hero({ title, subtitle, cta, image, className }: HeroProps) {
  return (
    <section
      className={cn(
        "relative overflow-hidden py-24 lg:py-32",
        className
      )}
    >
      <div className="container relative z-10">
        <div className="mx-auto max-w-3xl text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            {title}
          </h1>
          {subtitle && (
            <p className="mt-6 text-lg leading-8 text-muted-foreground">
              {subtitle}
            </p>
          )}
          {cta && <div className="mt-10 flex items-center justify-center gap-x-6">{cta}</div>}
        </div>
      </div>
      {image && (
        <div className="absolute inset-0 -z-10 opacity-20">
          <img
            src={image}
            alt=""
            className="h-full w-full object-cover"
          />
        </div>
      )}
    </section>
  )
}''',

        "Footer": '''import * as React from "react"
import Link from "next/link"
import { cn } from "@/lib/utils"

interface FooterLink {
  href: string
  label: string
}

interface FooterProps {
  links?: FooterLink[]
  copyright?: string
  className?: string
}

export function Footer({ links = [], copyright, className }: FooterProps) {
  return (
    <footer className={cn("border-t bg-background", className)}>
      <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            {copyright || `© ${new Date().getFullYear()} All rights reserved.`}
          </p>
        </div>
        {links.length > 0 && (
          <nav className="flex gap-4">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        )}
      </div>
    </footer>
  )
}''',
    }

    @classmethod
    def list_components(
        cls,
        category: Optional[ComponentCategory] = None,
    ) -> List[ComponentInfo]:
        """
        List available components.

        Args:
            category: Filter by category (optional)

        Returns:
            List of component info
        """
        components = list(cls.COMPONENTS.values())

        if category:
            components = [c for c in components if c.category == category]

        return sorted(components, key=lambda c: c.name)

    @classmethod
    def get_component(cls, name: str) -> Optional[str]:
        """
        Get component source code.

        Args:
            name: Component name

        Returns:
            Component TypeScript/React code or None
        """
        return cls.TEMPLATES.get(name)

    @classmethod
    def get_component_info(cls, name: str) -> Optional[ComponentInfo]:
        """
        Get component information.

        Args:
            name: Component name

        Returns:
            Component info or None
        """
        return cls.COMPONENTS.get(name)

    @classmethod
    def generate_component_file(
        cls,
        name: str,
        output_path: str = "components/ui",
    ) -> Dict[str, str]:
        """
        Generate component file for a project.

        Args:
            name: Component name
            output_path: Output directory path

        Returns:
            Dict of {filepath: content}
        """
        code = cls.get_component(name)
        if not code:
            return {}

        # Generate lowercase filename
        filename = name.lower().replace(" ", "-")
        filepath = f"{output_path}/{filename}.tsx"

        return {filepath: code}

    @classmethod
    def generate_all_components(
        cls,
        output_path: str = "components/ui",
    ) -> Dict[str, str]:
        """
        Generate all available component files.

        Args:
            output_path: Output directory path

        Returns:
            Dict of {filepath: content}
        """
        files = {}

        for name in cls.TEMPLATES:
            files.update(cls.generate_component_file(name, output_path))

        # Generate index file
        index_exports = []
        for name in sorted(cls.TEMPLATES.keys()):
            filename = name.lower().replace(" ", "-")
            index_exports.append(f'export * from "./{filename}"')

        files[f"{output_path}/index.ts"] = "\n".join(index_exports)

        return files


# Convenience functions
def get_component(name: str) -> Optional[str]:
    """Get component source code by name."""
    return ComponentLibrary.get_component(name)


def list_components(category: Optional[str] = None) -> List[ComponentInfo]:
    """List available components."""
    cat = ComponentCategory(category) if category else None
    return ComponentLibrary.list_components(cat)
