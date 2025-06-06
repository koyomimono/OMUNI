/* Aravis - Digital camera library
 *
 * Copyright © 2009-2016 Emmanuel Pacaud
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 * Author: Emmanuel Pacaud <emmanuel@gnome.org>
 */

#ifndef ARV_VERSION_H
#define ARV_VERSION_H

#if !defined (ARV_H_INSIDE) && !defined (ARAVIS_COMPILATION)
#error "Only <arv.h> can be included directly."
#endif

#include <arvtypes.h>

G_BEGIN_DECLS

/**
 * ARAVIS_VERSION:
 *
 * The version of the Aravis library.
 *
 * Since: 0.8.0
 */

#define ARAVIS_VERSION "0.8.26"

/**
 * ARAVIS_API_VERSION:
 *
 * The version of the Aravis library API.
 *
 * Since: 0.8.20
 */

#define ARAVIS_API_VERSION "0.8"

/**
 * ARAVIS_MAJOR_VERSION:
 *
 * The major version of the Aravis library.
 *
 * Since: 0.6.0
 */

#define ARAVIS_MAJOR_VERSION 0

/**
 * ARAVIS_MINOR_VERSION:
 *
 * The minor version of the Aravis library.
 *
 * Since: 0.6.0
 */

#define ARAVIS_MINOR_VERSION 8

/**
 * ARAVIS_MICRO_VERSION:
 *
 * The micor version of the Aravis library.
 *
 * Since: 0.6.0
 */

#define ARAVIS_MICRO_VERSION 26

/**
 * ARAVIS_CHECK_VERSION:
 * @major: the major version to check for
 * @minor: the minor version to check for
 * @micro: the micro version to check for
 *
 * Checks the version of the Aravis library that is being compiled
 * against.
 *
 * Returns: %TRUE if the version of the Aravis header files
 * is the same as or newer than the passed-in version.
 *
 * Since: 0.6.0
 */

#define ARAVIS_CHECK_VERSION(major,minor,micro) 				\
	(ARAVIS_MAJOR_VERSION > (major) ||					\
	 (ARAVIS_MAJOR_VERSION == (major) && ARAVIS_MINOR_VERSION > (minor)) ||	\
	 (ARAVIS_MAJOR_VERSION == (major) && ARAVIS_MINOR_VERSION == (minor) && \
	  ARAVIS_MICRO_VERSION >= (micro)))

G_END_DECLS

#endif
